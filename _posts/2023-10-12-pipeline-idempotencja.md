---
layout: default
title: "Pipeline, który można uruchomić drugi raz. Idempotencja w praktyce"
date: 2023-10-12 09:00:00 +0200
permalink: /blog/pipeline-idempotencja/
author: Rafał Prońko
categories: [data-engineering]
description: "Idempotentny zapis zamówień w Pythonie i SQLite: klucz, wersja, warunkowy UPSERT i transakcje."
image: /assets/images/pipeline-idempotencja.svg
---

# Pipeline, który można uruchomić drugi raz. Idempotencja w praktyce

![Schemat idempotentnego zapisu zamówień: partia z kluczem i wersją przechodzi przez warunkowy UPSERT, a retry nie tworzy duplikatów.]({{ '/assets/images/pipeline-idempotencja.svg' | relative_url }})

<time datetime="2023-10-12">12 października 2023</time>

Pipeline prędzej czy później zostanie uruchomiony drugi raz. Scheduler ponowi zadanie po timeoucie, ktoś ręcznie kliknie „retry”, backfill przejdzie po dniu, który już był załadowany.

Nie pytam więc tylko, czy import działa. Pytam, co po takim powtórzeniu zostanie w tabeli.

Jeśli pierwsze wykonanie zapisze zamówienia, a drugie dopisze ich kopie, mam problem. Jeśli powtórzenie starszej partii cofnie nowszą kwotę zamówienia, mam problem mniej widoczny, ale równie poważny. Zielony status zadania niewiele wtedy mówi o poprawności danych.

Chcę zbudować zapis, który mogę bezpiecznie powtórzyć. Zacznę od małego przykładu w Pythonie i SQLite — bez orkiestratora, brokera i dodatkowych bibliotek. Najpierw potrzebuję właściwej semantyki, dopiero potem narzędzi.

## Retry nie oznacza idempotencji

Retry to decyzja: „spróbuj jeszcze raz”. Idempotencja to właściwość operacji: ponowne wykonanie z tym samym wejściem nie zmienia jej końcowego efektu.

To rozróżnienie ma znaczenie zwłaszcza przy timeoutach. Brak odpowiedzi nie oznacza, że poprzednia próba niczego nie zapisała. Mogła zatwierdzić transakcję, a następnie stracić połączenie, zanim klient dostał potwierdzenie.

Dlatego nie chcę uzależniać poprawności od tego, czy potrafię odgadnąć wynik poprzedniej próby. Projektuję zapis tak, żeby powtórzenie było bezpieczne.

Pierwszy krok to ustalenie, **co identyfikuje ten sam obiekt biznesowy**.

Jeśli tabela zamówień ma tylko automatycznie nadawany identyfikator, każdy import może utworzyć nowy wiersz. Baza nie wie, że dwa rekordy opisują to samo zamówienie. Sama transakcja tego nie naprawi.

W przykładzie wybieram `order_id` z systemu źródłowego jako klucz główny. Jeśli łączę kilka źródeł z niezależnymi numeracjami, potrzebuję klucza złożonego, na przykład z identyfikatora źródła i zamówienia.

## Ten sam klucz nie zawsze oznacza duplikat

Zamówienie może wrócić z tą samą treścią, ponieważ ponowiłem import. Może też wrócić ze zmienioną kwotą, ponieważ system źródłowy je zaktualizował.

Odrzucenie każdego istniejącego `order_id` ochroni mnie przed duplikatami, ale zablokuje aktualizacje. Bezwarunkowe nadpisanie pozwoli aktualizować dane, lecz stworzy inny problem: starsza partia może cofnąć nowszy stan.

Potrzebuję więc informacji o kolejności zmian.

W tym przykładzie przyjmuję prosty kontrakt: każde zamówienie ma liczbową `version`, która rośnie przy kolejnych zmianach.

- Ten sam identyfikator i ta sama wersja oznaczają powtórzenie.
- Wyższa wersja oznacza aktualizację.
- Niższa wersja oznacza starsze dane, które nie powinny zastąpić nowszych.

To założenie o źródle, nie coś, co baza odgadnie za mnie. Jeśli źródło nie dostarcza wiarygodnej wersji, muszę zaprojektować inną regułę rozstrzygania konfliktów.

## Mały przykład: klucz, wersja i warunkowy UPSERT

Używam Pythona 3.11 i SQLite co najmniej 3.24.0. Wystarczy biblioteka standardowa.

Kwotę zapisuję jako całkowitą liczbę groszy. Dzięki temu nie wprowadzam do przykładu problemu reprezentacji pieniędzy przez liczby zmiennoprzecinkowe.

Baza działa w pamięci. Dwa ładowania wykonuję na tym samym połączeniu; zamknięcie połączenia usuwa dane. To demonstracja semantyki zapisu, nie trwały magazyn.

Poniższy kod można zapisać jako `pipeline.py` i uruchomić poleceniem `python3 pipeline.py`.

```python
import sqlite3
Order = tuple[str, int, int]  # (order_id, amount_cents, version)
SCHEMA = """CREATE TABLE orders (
    order_id TEXT PRIMARY KEY NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    version INTEGER NOT NULL)"""
UPSERT = """INSERT INTO orders VALUES (?, ?, ?)
    ON CONFLICT (order_id) DO UPDATE SET amount_cents = excluded.amount_cents,
    version = excluded.version WHERE excluded.version > orders.version"""


def load(con: sqlite3.Connection, batch: list[Order]) -> None:
    with con:  # commit albo rollback; połączenia nie zamyka
        con.executemany(UPSERT, batch)


def state(con: sqlite3.Connection) -> list[Order]:
    return con.execute("SELECT * FROM orders ORDER BY order_id").fetchall()


def check(ok: bool, name: str) -> None:
    if not ok:
        raise AssertionError(f"FAIL: {name}")
    print(f"PASS: {name}")


con = sqlite3.connect(":memory:")
try:
    con.execute(SCHEMA)
    batch = [("A-1", 1000, 1), ("A-2", 2500, 1)]
    load(con, batch)
    load(con, batch)
    check(state(con) == batch, "ponowny load partii daje identyczny stan")
    updated = [("A-1", 1200, 2), ("A-2", 2500, 1)]
    load(con, [("A-1", 1200, 2)])
    check(state(con) == updated, "wyższa version aktualizuje amount")
    load(con, [("A-1", 900, 1)])
    check(state(con) == updated, "starszy event nie cofa stanu")
    try:
        load(con, [("A-3", 500, 1), ("A-4", -1, 1)])
        raise AssertionError("FAIL: ujemny amount nie dał IntegrityError")
    except sqlite3.IntegrityError:
        check(state(con) == updated, "partia z błędem wycofana w całości")
finally:
    con.close()
```

Najważniejszy fragment to warunek:

```sql
WHERE excluded.version > orders.version
```

`excluded` oznacza rekord, który próbuję wstawić. `orders` oznacza rekord już zapisany.

Jeśli zamówienia nie ma, `INSERT` je tworzy. Jeśli istnieje, `ON CONFLICT` przechodzi do aktualizacji — ale tylko wtedy, gdy przychodząca wersja jest wyższa.

Powtórzenie tej samej wersji niczego nie zmienia. Starsze dane także nie zmieniają wiersza. Nowsza wersja aktualizuje kwotę i numer wersji.

Nie potrzebuję osobnego sprawdzania „czy rekord istnieje?” w Pythonie. Regułę egzekwuje baza w instrukcji zapisu.

## Transakcja rozwiązuje inny problem

W funkcji `load` cała partia jest objęta transakcją.

Przy domyślnej konfiguracji `sqlite3` w Pythonie 3.11 instrukcje zapisu otwierają transakcję. Blok `with con:` zatwierdza ją po poprawnym zakończeniu albo wycofuje, gdy wystąpi wyjątek. Nie zamyka połączenia — dlatego robię to osobno w `finally`.

W ostatnim teście pierwszy rekord partii jest poprawny, a drugi ma ujemną kwotę. Ograniczenie `CHECK` powoduje błąd. Po rollbacku w tabeli nie powinien zostać również pierwszy rekord tej partii.

To sprawdzenie **atomowości**, nie idempotencji.

Atomowość oznacza: partia zostaje zapisana cała albo wcale. Idempotencja oznacza: powtórzenie operacji nie zmienia końcowego efektu.

Bez atomowości mogę pozostawić po błędzie częściowo załadowane dane. Idempotentny zapis może pozwolić dokończyć import przez retry, ale odbiorcy przez pewien czas zobaczą stan pośredni.

Z kolei atomowa transakcja nie ochroni mnie przed duplikatami, jeśli dwukrotnie zatwierdzę partię opartą na zwykłym dopisywaniu wierszy.

Potrzebuję obu właściwości, ale z różnych powodów.

## Co właściwie sprawdzają testy?

Nie ograniczam się do liczby wierszy. Porównuję cały stan tabeli.

Sprawdzam, czy:

1. Powtórne ładowanie tej samej partii pozostawia identyczne dane.
2. Wyższa wersja aktualizuje kwotę zamówienia.
3. Starsze zdarzenie nie cofa aktualnego stanu.
4. Błąd w drugiej części partii wycofuje również wcześniejszy zapis z tej partii.

To niewielki zestaw, ale trafia w zachowania, na których opieram przykład.

Nie jest to test odporności na awarię procesu, współbieżnych zapisów czy problemów dyskowych. Takich gwarancji nie wyciągam z czterech sprawdzeń wykonywanych na bazie w pamięci.

## Gdzie kończy się ta gwarancja?

Pierwsza granica to kontrakt wersji.

Jeśli źródło wyśle tę samą wersję zamówienia z inną kwotą, przykład odrzuci zmianę bez alarmu. Tymczasem nie jest to zwykły duplikat — to sprzeczność danych.

W większym rozwiązaniu chciałbym ją wykrywać: przy równej wersji porównać treść i zgłosić konflikt. Warunkowy UPSERT nie zastępuje kontroli jakości źródła.

Druga granica to dostarczenie danych.

Idempotencja pozwala bezpiecznie przetwarzać powtórzenia. Nie sprawia, że brakująca wiadomość dotrze do systemu. Nie oznacza też, że zastosowałem każdą wersję pośrednią. W tym przykładzie interesuje mnie aktualny stan zamówienia, nie pełna historia jego zmian.

Trzecia granica to efekty poza bazą.

Jeśli podczas importu wyślę mail albo zlecę płatność, rollback transakcji nie cofnie tej operacji. Retry może wykonać ją ponownie. Klucz główny tabeli zamówień nie chroni skrzynki pocztowej ani rachunku bankowego.

W takim przypadku rozważam wzorzec **outbox**: razem ze zmianą danych zapisuję w jednej transakcji zamiar wykonania zewnętrznej operacji. Osobny proces odczytuje ten zamiar i realizuje go.

Ale outbox nie jest magicznym „exactly once”. Proces może wysłać wiadomość, a następnie przerwać działanie przed zapisaniem potwierdzenia. Ponowna próba wyśle ją jeszcze raz.

Potrzebuję więc także stabilnego identyfikatora operacji i deduplikacji po stronie odbiorcy, jeśli odbiorca ją obsługuje. Przy płatnościach sprawdzam warunki działania klucza idempotencji w konkretnym API, zamiast zakładać uniwersalną gwarancję.

## Od czego zaczynam projektowanie pipeline’u?

Nie od ustawienia liczby ponowień.

Najpierw ustalam, co identyfikuje rekord, jak rozpoznaję jego nowszą wersję i co powinno się wydarzyć po ponownym przetworzeniu tych samych danych.

Następnie określam granicę transakcji oraz sprawdzam częściowe niepowodzenie. Dopiero wtedy retry staje się mechanizmem odzyskiwania, a nie sposobem na zwielokrotnienie problemu.

W tym przykładzie gwarancja dotyczy stanu jednej tabeli i opiera się na poprawnym kontrakcie wersji. To ograniczenie, ale konkretne i możliwe do przetestowania.

Wolę taką gwarancję niż ogólne zapewnienie, że pipeline jest „odporny na błędy”.

### Źródła

- [SQLite: UPSERT](https://www.sqlite.org/lang_upsert.html) — składnia, `excluded`, warunkowa aktualizacja; UPSERT dostępny od SQLite 3.24.0 z 4 czerwca 2018 roku.
- [Python 3.11: sqlite3](https://docs.python.org/3.11/library/sqlite3.html) — transakcje i zachowanie context managera połączenia.

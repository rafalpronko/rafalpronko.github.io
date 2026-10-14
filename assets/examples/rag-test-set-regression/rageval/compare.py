from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pair:
    case: dict
    baseline: object
    candidate: object

    @property
    def case_id(self):
        return self.case["case_id"]

    @property
    def change(self):
        before, after = self.baseline.passed, self.candidate.passed
        if before == after:
            return "kept_pass" if after else "kept_fail"
        return "fixed" if after else "regressed"


@dataclass
class SliceStats:
    n: int = 0
    baseline_pass: int = 0
    candidate_pass: int = 0
    fixed: int = 0
    regressed: int = 0

    def add(self, pair):
        self.n += 1
        self.baseline_pass += pair.baseline.passed
        self.candidate_pass += pair.candidate.passed
        self.fixed += pair.change == "fixed"
        self.regressed += pair.change == "regressed"


@dataclass
class Comparison:
    pairs: list
    slices: dict = field(default_factory=dict)

    @property
    def regressions(self):
        return [p for p in self.pairs if p.change == "regressed"]

    @property
    def fixes(self):
        return [p for p in self.pairs if p.change == "fixed"]


def slice_keys(case):
    criticality = "critical" if case["critical"] else "noncritical"
    return [("all", "all"), ("criticality", criticality),
            ("split", case["split"]), ("kind", case["kind"])]


def compare(cases, baseline, candidate):
    """Pair baseline and candidate scores by case_id and aggregate per slice.
    Every rate uses all cases in the slice as its denominator."""
    result = Comparison([Pair(c, baseline[c["case_id"]], candidate[c["case_id"]]) for c in cases])
    for pair in result.pairs:
        for key in slice_keys(pair.case):
            result.slices.setdefault(key, SliceStats()).add(pair)
    return result

---
layout: default
---

# Blog

Piszę o ML, LLM i Data Engineering — od podstaw po praktyczne przykłady pracy z danymi i modelami.

## Artykuły

{% for post in site.posts %}
<article>
  {% if post.image %}
  <a href="{{ post.url | relative_url }}">
    <img src="{{ post.image | relative_url }}" alt="Ilustracja do artykułu: {{ post.title | escape }}" width="170" height="100" loading="lazy">
  </a>
  {% endif %}
  <p><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%d.%m.%Y" }}</time></p>
  <h3><a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
</article>
{% endfor %}

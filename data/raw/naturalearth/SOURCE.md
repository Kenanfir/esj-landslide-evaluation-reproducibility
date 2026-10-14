# Natural Earth country outlines (locator inset only)

**Source:** Natural Earth, 1:110m Cultural Vectors, Admin 0 – Countries.
Retrieved 2026-08-30 from the official Natural Earth vector repository,
`https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson`.

**Licence:** public domain. Natural Earth asks for no permission and imposes no
restrictions; attribution is courtesy, and the paper credits it in the caption
of the study-area figure.

**What is stored here:** `indonesia_ne110m.geojson` — the seven features needed
to draw the locator inset (Indonesia, Malaysia, Brunei, Timor-Leste,
Papua New Guinea, Australia, Philippines), extracted from the 839 KB world file
so the repository does not carry the whole world to draw one small map.

**Used by:** `scripts/generate_r1_figures.py`, Figure 1 inset only. It plays no
part in any analysis — no coordinate in the modelling table comes from it.

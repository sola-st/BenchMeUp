// Live parts of a page (elements with class "live" and an id) are re-fetched every
// data-refresh seconds while the "live" box is ticked. Open <details> stay open.
(function () {
  const secs = parseInt(document.body.dataset.refresh || "0", 10);
  const box = document.getElementById("live");
  const age = document.getElementById("live-age");
  if (!secs || !box) return;
  let last = Date.now();
  function openIds(root) {
    return new Set([...root.querySelectorAll("details[open][id]")].map(d => d.id));
  }
  async function tick() {
    if (!box.checked || document.hidden) return;
    try {
      const r = await fetch(location.href, { headers: { "X-Live": "1" } });
      if (!r.ok) return;
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      document.querySelectorAll(".live[id]").forEach(el => {
        const fresh = doc.getElementById(el.id);
        if (!fresh) return;
        const keep = openIds(el);
        el.innerHTML = fresh.innerHTML;
        keep.forEach(id => { const d = document.getElementById(id); if (d) d.open = true; });
      });
      last = Date.now();
    } catch (e) { /* the next tick tries again */ }
  }
  setInterval(tick, secs * 1000);
  setInterval(() => { if (age) age.textContent = "· " + Math.round((Date.now() - last) / 1000) + " s"; }, 1000);
})();

// Theme picker: auto (follow the system), light, dark, colorful; remembered in this browser.
(function () {
  const sel = document.getElementById("theme");
  if (!sel) return;
  sel.value = document.documentElement.dataset.theme || "auto";
  sel.addEventListener("change", () => {
    if (sel.value === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = sel.value;
    try { localStorage.setItem("bm-theme", sel.value); } catch (e) { /* not remembered */ }
  });
})();

// Client-side filter for tables: <input data-filter="table-id">
document.querySelectorAll("input[data-filter]").forEach(inp => {
  inp.addEventListener("input", () => {
    const q = inp.value.toLowerCase();
    const rows = document.querySelectorAll("#" + inp.dataset.filter + " tbody tr");
    let shown = 0;
    rows.forEach(tr => { const hit = tr.textContent.toLowerCase().includes(q); tr.hidden = !hit; shown += hit; });
    const out = document.getElementById(inp.dataset.filter + "-count");
    if (out) out.textContent = shown + " shown";
  });
});

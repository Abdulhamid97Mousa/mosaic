// Collapsible sidebar sections + sidebar toggle for Furo theme
document.addEventListener("DOMContentLoaded", function () {

  // ── Section fold/unfold ──
  var captions = document.querySelectorAll(".sidebar-tree p.caption");

  captions.forEach(function (captionP) {
    var list = captionP.nextElementSibling;
    if (!list || list.tagName !== "UL") return;

    var isActive = list.querySelector("a.current") !== null;

    if (!isActive) {
      captionP.classList.add("collapsed");
    }

    captionP.addEventListener("click", function () {
      captionP.classList.toggle("collapsed");
    });
  });

  // ── Sidebar show/hide ──

  // Place collapse button inside sidebar-brand (next to logo, right side)
  var sidebarBrand = document.querySelector(".sidebar-brand");
  if (!sidebarBrand) return;

  // Prevent the collapse button click from navigating (brand is an <a>)
  var collapseBtn = document.createElement("button");
  collapseBtn.className = "sidebar-collapse-btn";
  collapseBtn.setAttribute("aria-label", "Collapse sidebar");
  collapseBtn.setAttribute("title", "Collapse sidebar");
  // Left-pointing sidebar collapse icon (panel left)
  collapseBtn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="3" y="3" width="18" height="18" rx="2"/>' +
    '<line x1="9" y1="3" x2="9" y2="21"/>' +
    '<polyline points="15 9 12 12 15 15"/>' +
    '</svg>';
  sidebarBrand.style.position = "relative";
  sidebarBrand.appendChild(collapseBtn);

  // Expand button: clear icon in top-left when sidebar is gone
  var expandBtn = document.createElement("button");
  expandBtn.className = "sidebar-expand-btn";
  expandBtn.setAttribute("aria-label", "Expand sidebar");
  expandBtn.setAttribute("title", "Expand sidebar");
  // Right-pointing sidebar expand icon (panel left open)
  expandBtn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="3" y="3" width="18" height="18" rx="2"/>' +
    '<line x1="9" y1="3" x2="9" y2="21"/>' +
    '<polyline points="13 9 16 12 13 15"/>' +
    '</svg>';
  document.body.appendChild(expandBtn);

  // Restore state
  if (localStorage.getItem("mosaic-sidebar-hidden") === "true") {
    document.body.classList.add("sidebar-hidden");
  }

  collapseBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    document.body.classList.add("sidebar-hidden");
    localStorage.setItem("mosaic-sidebar-hidden", "true");
  });

  expandBtn.addEventListener("click", function () {
    document.body.classList.remove("sidebar-hidden");
    localStorage.setItem("mosaic-sidebar-hidden", "false");
  });
});

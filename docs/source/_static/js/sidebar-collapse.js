// Collapsible sidebar sections for Furo theme
document.addEventListener("DOMContentLoaded", function () {
  // Target <p class="caption"> elements (the section headers)
  var captions = document.querySelectorAll(".sidebar-tree p.caption");

  captions.forEach(function (captionP) {
    var list = captionP.nextElementSibling;
    if (!list || list.tagName !== "UL") return;

    // Check if this section contains the current page
    var isActive = list.querySelector("a.current") !== null;

    // Collapse sections that don't contain the current page
    if (!isActive) {
      captionP.classList.add("collapsed");
    }

    // Toggle on click anywhere on the caption <p>
    captionP.addEventListener("click", function () {
      captionP.classList.toggle("collapsed");
    });
  });
});

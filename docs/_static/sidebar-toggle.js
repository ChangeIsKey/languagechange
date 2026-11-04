document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".sphinxsidebar li").forEach((item, index) => {
    const children = Array.from(item.children);
    const submenu = children.find(element => element.tagName === "UL");
    const link = children.find(element => element.tagName === "A");

    // Add arrows only to headings that have children.
    if (!submenu || !link) return;

    item.classList.add("has-toc-toggle");
    submenu.id = submenu.id || `sidebar-section-${index}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "toc-toggle";
    button.setAttribute("aria-controls", submenu.id);

    function setExpanded(expanded) {
      submenu.hidden = !expanded;
      button.setAttribute("aria-expanded", String(expanded));

      const action = expanded ? "Collapse" : "Expand";
      button.setAttribute("aria-label", `${action} ${link.textContent.trim()}`);
      button.title = `${action} section`;
    }

    button.addEventListener("click", () => {
      setExpanded(submenu.hidden);
    });

    item.insertBefore(button, link);

    // Keep the current page and any directly linked section visible.
    const containsTarget = window.location.hash &&
      Array.from(submenu.querySelectorAll("a")).some(
        anchor => anchor.getAttribute("href") === window.location.hash
      );

    setExpanded(item.classList.contains("current") || Boolean(containsTarget));
  });
});

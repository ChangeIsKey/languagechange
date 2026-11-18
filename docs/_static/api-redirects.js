/* Preserve bookmarks to objects moved from older combined API pages. */
(() => {
  const pages = {
  "languagechange.html": [
    "languagechange.corpora",
    "languagechange.search",
    "languagechange.usages",
    "languagechange.benchmark",
    "languagechange.cache",
    "languagechange.resource_manager",
    "languagechange.pipeline",
    "languagechange.visualization",
    "languagechange.utils"
  ],
  "languagechange.models.representation.html": [
    "languagechange.models.representation.static",
    "languagechange.models.representation.contextualized",
    "languagechange.models.representation.definition",
    "languagechange.models.representation.prompting",
    "languagechange.models.representation.alignment"
  ],
  "languagechange.models.meaning.html": [
    "languagechange.models.meaning.clustering",
    "languagechange.models.meaning.meaning"
  ],
  "languagechange.models.change.html": [
    "languagechange.models.change.metrics",
    "languagechange.models.change.timeseries",
    "languagechange.models.change.widid"
  ],
  "languagechange.models.type.html": [
    "languagechange.models.type.type_classification"
  ]
};
  const page = window.location.pathname.split('/').pop();
  if (!pages[page] || !window.location.hash) return;
  let fragment;
  try { fragment = decodeURIComponent(window.location.hash.slice(1)); }
  catch { return; }
  for (const name of pages[page]) {
    const oldHeading = name.replace(/[._]/g, '-') + '-module';
    if (fragment === 'module-' + name || fragment === oldHeading ||
        fragment === name || fragment.startsWith(name + '.')) {
      const destination = new URL(name + '.html', window.location.href);
      destination.hash = window.location.hash;
      window.location.replace(destination.href);
      return;
    }
  }
})();

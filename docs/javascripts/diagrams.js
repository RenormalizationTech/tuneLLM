// SuperFences emits <pre class="mermaid"><code>...</code></pre>. Mermaid.run
// reads innerHTML, so remove that wrapper using decoded textContent first.
function restoreDiagramSource(node, source, error) {
  node.textContent = source; // Also removes a partially generated/error SVG.
  node.removeAttribute("data-processed");
  node.dataset.mermaidState = "source";
  const note = document.createElement("p");
  note.className = "diagram-fallback-note";
  note.textContent = "Diagram renderer unavailable; original source is shown below.";
  node.before(note);
  console.error("Mermaid rendering unavailable", error);
}

async function renderMermaidDiagrams(nodes, mermaid) {
  for (const node of nodes) {
    const source = node.querySelector("code")?.textContent ?? node.textContent;
    node.textContent = source;
    try {
      await mermaid.run({ nodes: [node] });
      if (!node.querySelector("svg") || node.querySelector(".error-icon, .error-text")) {
        throw new Error("Mermaid did not produce a successful diagram");
      }
      node.dataset.mermaidState = "rendered";
    } catch (error) {
      restoreDiagramSource(node, source, error);
    }
  }
}

// Pinned module import; failed/offline loading restores original diagram text.
document.addEventListener("DOMContentLoaded", async () => {
  const nodes = document.querySelectorAll(".mermaid");
  if (!nodes.length) return;
  try {
    const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs");
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
    await renderMermaidDiagrams(nodes, mermaid);
  } catch (error) {
    for (const node of nodes) {
      const source = node.querySelector("code")?.textContent ?? node.textContent;
      restoreDiagramSource(node, source, error);
    }
  }
});

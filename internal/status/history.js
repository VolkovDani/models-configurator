const form = document.querySelector("#more-calls");

if (form && typeof IntersectionObserver !== "undefined") {
  const calls = document.querySelector("#calls");
  const feedback = document.querySelector("#history-feedback");
  const button = form.querySelector("button");
  const cursor = form.elements.before;
  let loading = false;

  async function loadOlder() {
    if (loading || !cursor.value) return;
    loading = true;
    observer.unobserve(form);
    button.setAttribute("aria-disabled", "true");

    try {
      const url = new URL(form.action);
      url.search = new URLSearchParams(new FormData(form)).toString();
      const response = await fetch(url);
      if (!response.ok) throw new Error();
      const page = new DOMParser().parseFromString(await response.text(), "text/html");
      const rows = page.querySelectorAll("#calls tr");
      for (const row of rows) calls.append(row);

      cursor.value = page.querySelector("#more-calls [name=before]")?.value ?? "";
      feedback.textContent = cursor.value ? form.dataset.loaded : form.dataset.end;
      if (cursor.value) observer.observe(form);
    } catch {
      feedback.textContent = form.dataset.error;
    } finally {
      loading = false;
      if (cursor.value) button.removeAttribute("aria-disabled");
    }
  }

  const observer = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) loadOlder();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadOlder();
  });
  observer.observe(form);
}

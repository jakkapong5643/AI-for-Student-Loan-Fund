export async function askApi(question: string) {
  const r = await fetch(process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8001/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: "web-user" })
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

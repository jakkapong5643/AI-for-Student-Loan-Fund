interface ChatBubbleProps {
  role: "user" | "assistant";
  text: string;
}

export default function ChatBubble({ role, text }: ChatBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`p-3 rounded-2xl max-w-[75%] shadow text-sm whitespace-pre-wrap
        ${isUser ? "bg-blue-600 text-white" : "bg-white border border-gray-200 text-gray-800"}`}
      >
        {text}
      </div>
    </div>
  );
}

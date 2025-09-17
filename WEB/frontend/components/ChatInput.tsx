"use client";

import { useState } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSend = () => {
    if (!value.trim()) return;
    onSend(value);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex gap-3 items-end">
      {/* Textarea */}
      <textarea
  className="flex-1 resize-none rounded-2xl border border-gray-200 px-4 py-3 
             bg-white/70 backdrop-blur-md shadow-sm text-black
             focus:outline-none focus:ring-2 focus:ring-green-500/60 
             transition"
  rows={2}
  placeholder="พิมพ์คำถามที่นี่…"
  value={value}
  onChange={(e) => setValue(e.target.value)}
  onKeyDown={handleKeyDown}
  disabled={disabled}
/>

      <button
        onClick={handleSend}
        disabled={disabled}
        className="shrink-0 rounded-2xl bg-gradient-to-r from-green-600 to-blue-700 
                   hover:from-green-500 hover:to-blue-600
                   text-white font-semibold px-5 py-3 shadow-md 
                   hover:shadow-green-300/40 hover:scale-105
                   transition disabled:opacity-50 disabled:hover:scale-100"
      >
        ส่ง ➤
      </button>
    </div>
  );
}

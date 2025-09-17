"use client";

import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-md">
      <div className="mx-auto max-w-6xl px-4 py-3 flex justify-between items-center">
        {/* Logo */}
        <Link
          href="/"
          className="text-xl font-extrabold bg-gradient-to-r from-green-600 to-blue-700 bg-clip-text text-transparent"
        >
          กยศ. Chatbot
        </Link>

        {/* Menu */}
        <div className="flex gap-8 text-sm font-semibold">
          <Link
            href="/"
            className="transition transform hover:scale-110 hover:text-green-600"
          >
            Home
          </Link>
          <Link
            href="/chat"
            className="transition transform hover:scale-110 hover:text-green-600"
          >
            Chat
          </Link>
          <Link
            href="/about"
            className="transition transform hover:scale-110 hover:text-green-600"
          >
            About
          </Link>
        </div>
      </div>
    </nav>
  );
}

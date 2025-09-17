import "./globals.css";
import Navbar from "../components/Navbar";

export const metadata = {
  title: "RAG Chatbot กยศ.",
  viewport: {
    width: "device-width",
    initialScale: 1.2, 
    maximumScale: 1.2,
  },
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th">
      <body className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
        <Navbar />
        {children}
      </body>
    </html>
  );
}

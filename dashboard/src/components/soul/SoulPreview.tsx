import ReactMarkdown from "react-markdown";

interface SoulPreviewProps {
  content: string;
}

export function SoulPreview({ content }: SoulPreviewProps) {
  if (!content.trim()) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-600">
        Preview will appear here...
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="prose prose-invert prose-sm max-w-none prose-headings:text-gray-100 prose-p:text-gray-300 prose-a:text-indigo-400 prose-code:text-indigo-300 prose-strong:text-gray-200">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

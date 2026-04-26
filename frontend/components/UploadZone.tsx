"use client";

import { useCallback, useState } from "react";
import { Upload, ImageIcon } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function UploadZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setPreview(URL.createObjectURL(file));
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <label
      className={`relative flex flex-col items-center justify-center w-full rounded-xl border-2 border-dashed transition-colors cursor-pointer overflow-hidden
        ${disabled ? "opacity-50 pointer-events-none" : ""}
        ${dragging ? "border-violet-400 bg-violet-950/30" : "border-white/10 bg-white/[0.03] hover:border-white/20"}`}
      style={{ minHeight: 200 }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        disabled={disabled}
      />

      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="preview" className="w-full h-full object-contain max-h-64" />
      ) : (
        <div className="flex flex-col items-center gap-3 py-10 px-6 text-center">
          <div className="rounded-full bg-white/5 p-4">
            {dragging ? (
              <ImageIcon className="w-7 h-7 text-violet-400" />
            ) : (
              <Upload className="w-7 h-7 text-white/40" />
            )}
          </div>
          <p className="text-sm text-white/50">
            Drop an image or <span className="text-violet-400 underline">browse</span>
          </p>
          <p className="text-xs text-white/25">JPG, PNG, WEBP</p>
        </div>
      )}
    </label>
  );
}

import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, Square } from "lucide-react";
import { cn } from "@/lib/utils";

interface VoiceInputProps {
  onVoiceData: (audio: string, mimeType: string) => void;
  disabled?: boolean;
}

const MAX_DURATION = 30_000; // 30 seconds auto-stop

export function VoiceInput({ onVoiceData, disabled = false }: VoiceInputProps) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const autoStopRef = useRef<ReturnType<typeof setTimeout>>();

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (autoStopRef.current) clearTimeout(autoStopRef.current);
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      chunksRef.current = [];
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        // Stop all tracks
        stream.getTracks().forEach((t) => t.stop());

        // Clear timers
        if (timerRef.current) clearInterval(timerRef.current);
        if (autoStopRef.current) clearTimeout(autoStopRef.current);

        // Convert to base64 and send
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result as string;
          // Extract base64 data after the data URL prefix
          const base64 = result.split(",")[1];
          if (base64) {
            onVoiceData(base64, "audio/webm");
          }
        };
        reader.readAsDataURL(blob);

        setRecording(false);
        setDuration(0);
      };

      mediaRecorder.start(250); // Collect data every 250ms
      setRecording(true);
      setDuration(0);

      // Duration counter
      timerRef.current = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);

      // Auto-stop after MAX_DURATION
      autoStopRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      }, MAX_DURATION);
    } catch {
      // Permission denied or not supported
      setRecording(false);
    }
  }, [onVoiceData]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const handleClick = useCallback(() => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  const formatDuration = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${String(secs).padStart(2, "0")}`;
  };

  if (recording) {
    return (
      <div className="flex items-center gap-2">
        <span className="flex items-center gap-1.5 text-xs text-red-400">
          <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
          {formatDuration(duration)}
        </span>
        <button
          type="button"
          onClick={handleClick}
          className={cn(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
            "bg-red-600 text-white transition-colors",
            "hover:bg-red-500 active:bg-red-700",
          )}
        >
          <Square className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled}
      className={cn(
        "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
        "bg-gray-700 text-gray-300 transition-colors",
        "hover:bg-gray-600 hover:text-white active:bg-gray-800",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-gray-700",
      )}
    >
      <Mic className="h-4 w-4" />
    </button>
  );
}

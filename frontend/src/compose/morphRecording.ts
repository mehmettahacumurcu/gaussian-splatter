export const DEFAULT_RECORDING_FPS = 30;

/** Include both endpoints; render time never changes these timeline samples. */
export function createFrameSchedule(duration: number, fps = DEFAULT_RECORDING_FPS): number[] {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(fps) || fps <= 0) {
    throw new RangeError("Recording duration and fps must be positive finite numbers");
  }
  const intervals = Math.round(duration * fps);
  if (!Number.isSafeInteger(intervals) || intervals < 1) {
    throw new RangeError("Recording must contain at least one finite frame interval");
  }
  return Array.from({ length: intervals + 1 }, (_, index) => index / intervals);
}

export function selectRecordingMimeType(isSupported: (mime: string) => boolean): string | null {
  return ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(isSupported) ?? null;
}

export interface MorphRecordingOptions {
  canvas: HTMLCanvasElement;
  duration: number;
  fps?: number;
  signal: AbortSignal;
  /** Fraction of scheduled frames captured, from 0 to 1. */
  onProgress?: (progress: number) => void;
  /** Prepare/sort t, await waitForCapture(), then render and call capture synchronously. */
  renderFrame: (t: number, signal: AbortSignal, capture: () => void, waitForCapture: () => Promise<void>) => Promise<void>;
}

function abortError(): DOMException {
  return new DOMException("Recording cancelled", "AbortError");
}

function checkSignal(signal: AbortSignal): void {
  if (signal.aborted) throw signal.reason ?? abortError();
}

/** Race even render callbacks which do not cooperate with cancellation. */
function abortable<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason ?? abortError());
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
    operation.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    checkSignal(signal);
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason ?? abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, Math.max(0, Math.ceil(ms)));
    signal.addEventListener("abort", abort, { once: true });
  });
}

function afterPaint(signal: AbortSignal): Promise<void> {
  if (typeof requestAnimationFrame !== "function") return delay(0, signal);
  return new Promise((resolve, reject) => {
    checkSignal(signal);
    const abort = () => {
      cancelAnimationFrame(frame);
      reject(signal.reason ?? abortError());
    };
    const frame = requestAnimationFrame(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    });
    signal.addEventListener("abort", abort, { once: true });
  });
}

/**
 * Manual capture preserves every fixed timeline sample. MediaRecorder supplies
 * real-time timestamps, so slow rendering/sorting can lengthen the WebM. The
 * frame interval is a minimum, not permission to skip late samples.
 */
export async function recordMorphVideo({
  canvas, duration, fps = DEFAULT_RECORDING_FPS, signal, onProgress, renderFrame,
}: MorphRecordingOptions): Promise<Blob | null> {
  if (signal.aborted) return null;
  const schedule = createFrameSchedule(duration, fps);
  const initialWidth = canvas.width;
  const initialHeight = canvas.height;
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function" ||
      typeof canvas.captureStream !== "function") {
    throw new Error("Bu tarayıcı video kaydını desteklemiyor (MediaRecorder / captureStream).");
  }
  const mimeType = selectRecordingMimeType((mime) => MediaRecorder.isTypeSupported(mime));
  if (!mimeType) throw new Error("Bu tarayıcı WebM video kaydını desteklemiyor.");

  const controller = new AbortController();
  const workingSignal = controller.signal;
  let stream: MediaStream | undefined;
  let recorder: MediaRecorder | undefined;
  let track: CanvasCaptureMediaStreamTrack | undefined;
  let cancelled = false;
  let stopRequested = false;
  const chunks: BlobPart[] = [];
  const stopTracks = () => {
    for (const track of stream?.getTracks() ?? []) {
      try { track.stop(); } catch { /* Keep releasing the other tracks. */ }
    }
  };
  const stopSafely = () => {
    if (recorder && !stopRequested && recorder.state !== "inactive") {
      stopRequested = true;
      try { recorder.stop(); } catch { /* Cleanup must preserve the original failure/cancel. */ }
    }
  };
  const cancel = () => {
    cancelled = true;
    controller.abort(abortError());
    stopSafely();
    stopTracks();
  };
  const failed = (event: Event) => {
    const cause = (event as Event & { error?: DOMException }).error;
    controller.abort(cause ?? new Error("Video kaydı başarısız oldu."));
    stopSafely();
    stopTracks();
  };
  let resolveStopped: () => void = () => {};
  const stopped = new Promise<void>((resolve) => { resolveStopped = resolve; });
  const didStop = () => {
    resolveStopped();
    if (!stopRequested && !workingSignal.aborted) {
      controller.abort(new Error("Video kaydı beklenmedik şekilde durdu."));
      stopTracks();
    }
  };
  const data = (event: BlobEvent) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  signal.addEventListener("abort", cancel, { once: true });
  if (signal.aborted) cancel();

  try {
    checkSignal(workingSignal);
    onProgress?.(0);

    let lastCapture = -Infinity;
    const interval = 1000 / fps;
    for (let index = 0; index < schedule.length; index++) {
      checkSignal(workingSignal);
      let captured = false;
      let acceptingCapture = true;
      let captureReady = index === 0;
      const waitForCapture = async () => {
        checkSignal(workingSignal);
        if (index > 0) await delay(interval - (performance.now() - lastCapture), workingSignal);
        checkSignal(workingSignal);
        captureReady = true;
      };
      try {
        await abortable(renderFrame(schedule[index], workingSignal, () => {
          checkSignal(workingSignal);
          if (!acceptingCapture || captured) throw new Error("Recording render must capture exactly one frame");
          if (!captureReady) throw new Error("Recording render must await capture pacing before its final render");
          if (canvas.width !== initialWidth || canvas.height !== initialHeight) {
            throw new Error("Kayıt sırasında görünüm boyutu değişti. Lütfen yeniden kaydedin.");
          }
          if (index === 0) {
            // captureStream requests an initial frame even at 0 fps. Creating
            // the stream only after the prepared draw prevents a stale first frame.
            stream = canvas.captureStream(0);
            checkSignal(workingSignal);
            track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined;
            if (!track || typeof track.requestFrame !== "function") {
              throw new Error("Bu tarayıcı kare kontrollü video kaydını desteklemiyor (requestFrame).");
            }
            recorder = new MediaRecorder(stream, { mimeType });
            recorder.addEventListener("dataavailable", data);
            recorder.addEventListener("error", failed);
            recorder.addEventListener("stop", didStop);
            checkSignal(workingSignal);
            recorder.start();
          }
          track!.requestFrame();
          captured = true;
          lastCapture = performance.now();
        }, waitForCapture), workingSignal);
      } finally {
        acceptingCapture = false;
      }
      checkSignal(workingSignal);
      if (!captured) throw new Error("Recording render did not capture its frame");
      onProgress?.((index + 1) / schedule.length);
    }

    // Give the final request a paint/encoding opportunity before finalizing.
    await delay(interval, workingSignal);
    await afterPaint(workingSignal);
    checkSignal(workingSignal);
    if (!recorder) throw new Error("Video kaydı başlatılamadı.");
    stopRequested = true;
    recorder.stop();
    await abortable(stopped, workingSignal);
    checkSignal(workingSignal);
    if (chunks.length === 0) throw new Error("Video kaydı boş çıktı; lütfen yeniden deneyin.");
    return new Blob(chunks, { type: recorder.mimeType || mimeType });
  } catch (error) {
    if (cancelled || signal.aborted) return null;
    throw workingSignal.aborted ? workingSignal.reason : error;
  } finally {
    signal.removeEventListener("abort", cancel);
    stopSafely();
    stopTracks();
    recorder?.removeEventListener("dataavailable", data);
    recorder?.removeEventListener("error", failed);
    recorder?.removeEventListener("stop", didStop);
  }
}

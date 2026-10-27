"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  describeFace,
  countFaces,
  fileToImage,
  loadModels,
  NoFaceError,
} from "@/lib/faceapi";
import { rankMatches, type CelebrityDataset, type Match } from "@/lib/match";

const STAGES = [
  "Starting up the ML backend",
  "Loading face detector",
  "Loading landmark model",
  "Loading recognition model",
  "Ready",
];

interface Shot {
  url: string;
  width: number;
  height: number;
  box: { x: number; y: number; width: number; height: number };
  faces: number;
}

export default function MatchApp() {
  const [dataset, setDataset] = useState<CelebrityDataset | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [stageLabel, setStageLabel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shot, setShot] = useState<Shot | null>(null);
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const shotUrlRef = useRef<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Pull the celebrity descriptors. Served as a static file so the dataset can
  // be swapped by replacing one file, without touching the bundle.
  useEffect(() => {
    let cancelled = false;
    fetch("/celebrities.json")
      .then((res) => {
        if (!res.ok) throw new Error(`celebrities.json returned ${res.status}`);
        return res.json();
      })
      .then((json: CelebrityDataset) => {
        if (!cancelled) setDataset(json);
      })
      .catch((err: Error) => {
        if (!cancelled) setDatasetError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (shotUrlRef.current) URL.revokeObjectURL(shotUrlRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  // The <video> element only exists once the camera panel renders, so the
  // stream is attached here rather than inside openCamera.
  useEffect(() => {
    const video = videoRef.current;
    if (cameraOpen && video && streamRef.current) {
      video.srcObject = streamRef.current;
      void video.play().catch(() => {});
    }
  }, [cameraOpen]);

  const stageIndex = stageLabel ? STAGES.indexOf(stageLabel) : -1;
  const stagePercent = stageIndex < 0 ? 0 : ((stageIndex + 1) / STAGES.length) * 100;

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setMatches(null);
      setShot(null);
      setBusy(true);

      try {
        await loadModels(setStageLabel);
        setStageLabel("Finding a face");

        const img = await fileToImage(file);
        const [face, faces] = await Promise.all([describeFace(img), countFaces(img)]);

        if (shotUrlRef.current) URL.revokeObjectURL(shotUrlRef.current);
        const url = URL.createObjectURL(file);
        shotUrlRef.current = url;

        setShot({
          url,
          width: img.naturalWidth,
          height: img.naturalHeight,
          box: face.box,
          faces,
        });

        const pool = dataset?.celebrities ?? [];
        setMatches(pool.length > 0 ? rankMatches(face.descriptor, pool, 5) : []);
      } catch (err) {
        if (err instanceof NoFaceError) {
          setError(
            "No face found in that picture. A clear, front-facing photo with the face reasonably large in frame works best.",
          );
        } else {
          setError(err instanceof Error ? err.message : "Something went wrong reading that image.");
        }
      } finally {
        setBusy(false);
        setStageLabel(null);
      }
    },
    [dataset],
  );

  const onPick = () => inputRef.current?.click();

  const closeCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }, []);

  const openCamera = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser can't open a camera here. Try choosing a photo instead.");
      return;
    }
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      setCameraOpen(true);
    } catch (err) {
      const denied = err instanceof DOMException && err.name === "NotAllowedError";
      setError(
        denied
          ? "Camera permission was blocked. Allow camera access in your browser settings, or choose a photo instead."
          : "Couldn't start the camera. Check that one is connected and not in use by another app.",
      );
    }
  };

  // Grabs the current video frame, mirrored to match the selfie preview, and
  // feeds it through the same path as an uploaded file.
  const capture = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        closeCamera();
        if (blob) void handleFile(new File([blob], "camera.jpg", { type: "image/jpeg" }));
      },
      "image/jpeg",
      0.92,
    );
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  };

  const reset = () => {
    setShot(null);
    setMatches(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const celebCount = dataset?.celebrities.length ?? 0;
  const hasDataset = celebCount > 0;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <h1 className="masthead">
            Which Bollywood
            <span className="line-star">star</span>
            is your face?
          </h1>
          <p className="standfirst">
            Drop in a photo or take a selfie. Your face gets turned into 128 numbers, and those numbers get
            compared against every star in the reference set.
          </p>
        </div>

        <div className="plate">
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="visually-hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
            }}
          />

          {cameraOpen ? (
            <div className="camera">
              <video ref={videoRef} className="camera-feed" playsInline muted />
              <div className="camera-actions">
                <button type="button" className="btn" onClick={capture}>
                  Click photo
                </button>
                <button type="button" className="btn btn-outline" onClick={closeCamera}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
          <>
          <button
            type="button"
            className={`dropzone${dragOver ? " is-over" : ""}`}
            disabled={busy}
            onClick={onPick}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <p className="dropzone-head">{busy ? "Working…" : "Choose a photo"}</p>
            <p className="dropzone-sub">
              {busy ? "Hold on a moment" : "or drag one onto this panel"}
            </p>
          </button>

          <button
            type="button"
            className="btn btn-outline camera-open"
            disabled={busy}
            onClick={() => void openCamera()}
          >
            Open camera
          </button>
          </>
          )}

          {busy && stageLabel && (
            <div className="progress">
              <div className="progress-label">
                <span>{stageLabel}</span>
                <span>{Math.round(stagePercent)}%</span>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-valuenow={Math.round(stagePercent)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Model loading progress"
              >
                <div className="progress-fill" style={{ width: `${stagePercent}%` }} />
              </div>
            </div>
          )}

          {error && <p className="notice">{error}</p>}

          {!busy && !error && (
            <p className="plate-foot">
              First run downloads about 12&nbsp;MB of model weights, then caches them. Everything
              after that is instant.
            </p>
          )}
        </div>

        <div className="credit-strip">
          <span>
            Runs in <b>your browser</b> — the photo is never uploaded
          </span>
          <span>
            Reference set: <b>{hasDataset ? `${celebCount} stars` : "not loaded yet"}</b>
          </span>
          <span>
            Embedding: <b>128 dimensions</b>, ranked by euclidean distance
          </span>
        </div>
      </section>

      {shot && (
        <section className="results" aria-live="polite">
          <div className="verdict">
            <figure className="shot">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={shot.url} alt="The photo you chose" />
              <span
                className="shot-box"
                style={{
                  left: `${(shot.box.x / shot.width) * 100}%`,
                  top: `${(shot.box.y / shot.height) * 100}%`,
                  width: `${(shot.box.width / shot.width) * 100}%`,
                  height: `${(shot.box.height / shot.height) * 100}%`,
                  margin: 10,
                }}
              />
            </figure>

            <div>
              {matches && matches.length > 0 ? (
                <>
                  <p className="verdict-rank">Closest match</p>
                  <h2 className="verdict-name">{matches[0].celebrity.name}</h2>
                  <p className="verdict-score">
                    <strong>{matches[0].percent}%</strong>
                    <span>a {matches[0].closeness} match</span>
                  </p>
                  <div className="meter">
                    <div className="meter-fill" style={{ width: `${matches[0].percent}%` }} />
                  </div>
                  <p className="verdict-meta">
                    Euclidean distance {matches[0].distance.toFixed(3)} across{" "}
                    {matches[0].celebrity.samples} reference photos — anything under 0.6 is where
                    face recognition would call two photos the same person. Cosine similarity{" "}
                    {matches[0].similarity.toFixed(3)}. This measures how close two face
                    embeddings sit, which is not the same as looking alike to a person.
                  </p>
                </>
              ) : (
                <>
                  <p className="verdict-rank">Face found</p>
                  <h2 className="verdict-name">128 numbers, no one to compare</h2>
                  <p className="verdict-meta">
                    Detection and embedding both worked — the pipeline is running correctly in
                    your browser. There is just no reference set loaded to match against yet.
                    Build one with the steps below and this turns into a real verdict.
                  </p>
                </>
              )}

              {shot.faces > 1 && (
                <p className="notice on-ink">
                  {shot.faces} faces in this photo. Only the most prominent one was matched — crop
                  to a single person for a cleaner read.
                </p>
              )}

              <div className="actions">
                <button type="button" className="btn" onClick={onPick}>
                  Try another photo
                </button>
                <button
                  type="button"
                  className="btn btn-quiet"
                  onClick={() => {
                    window.scrollTo({ top: 0, behavior: "smooth" });
                    void openCamera();
                  }}
                >
                  Use camera
                </button>
                <button type="button" className="btn btn-quiet" onClick={reset}>
                  Clear
                </button>
              </div>
            </div>
          </div>

          {matches && matches.length > 1 && (
            <div className="runners">
              <h3>Also close</h3>
              <ol className="runner-list">
                {matches.slice(1).map((match, i) => (
                  <li className="runner" key={match.celebrity.slug}>
                    <span className="runner-pos">{i + 2}</span>
                    <span className="runner-name">
                      {match.celebrity.name}
                      <span className="runner-samples">
                        {match.closeness} · distance {match.distance.toFixed(3)}
                      </span>
                    </span>
                    <span className="runner-pct">{match.percent}%</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>
      )}

      {!hasDataset && (
        <section className="setup">
          <h2>Load a reference set</h2>
          <p>
            {datasetError
              ? `No reference data yet (${datasetError}).`
              : "No reference data yet."}{" "}
            The app ships without celebrity photos on purpose — those images are other people&rsquo;s
            copyright and likeness. You point the build script at a dataset you have, and it writes
            out nothing but numbers.
          </p>
          <ol>
            <li>
              Download the Bollywood celebrity faces dataset from Kaggle and unzip it. You want one
              folder per person, each holding that person&rsquo;s photos.
            </li>
            <li>
              Install what the script needs, once: <code>npm run embeddings:setup</code>
            </li>
            <li>
              Generate the embeddings:{" "}
              <code>npm run embeddings -- --input ./path/to/dataset</code>
            </li>
            <li>
              That writes <code>public/celebrities.json</code>. Redeploy and this page starts
              matching.
            </li>
          </ol>
        </section>
      )}

      <footer className="colophon">
        <p>
          Face detection uses SSD MobileNet v1; the 128-dimension embeddings come from a ResNet
          recognition network, both via face-api running on TensorFlow.js. Ranking is euclidean
          distance against the mean embedding of each person&rsquo;s reference photos.
        </p>
        <p>
          Built as a browser-side rewrite of the CampusX &ldquo;Which Bollywood celebrity are
          you?&rdquo; project, which does the same thing in Python with VGGFace and Streamlit. A
          lookalike score is a toy, not an identity check — please don&rsquo;t use it as one.
        </p>
      </footer>
    </main>
  );
}

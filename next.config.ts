import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Nothing to tweak in the bundler: the browser code imports
  // @vladmandic/face-api/dist/face-api.esm.js directly, so the package's Node
  // entrypoint (which wants @tensorflow/tfjs-node) is never pulled in.
  turbopack: {},

  // Keep the heavy, browser-only bits out of the serverless function bundle.
  outputFileTracingExcludes: {
    "*": [
      "./public/models/**",
      "./node_modules/@tensorflow/**",
      "./node_modules/sharp/**",
      "./node_modules/@vladmandic/**",
    ],
  },

  async headers() {
    return [
      {
        // Model weights are immutable and large — let the browser keep them.
        source: "/models/:path*",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
      },
    ];
  },
};

export default nextConfig;

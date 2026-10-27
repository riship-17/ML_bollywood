# Which Bollywood star is your face?

Upload a photo, get your closest Bollywood lookalike. Face detection, embedding and
matching all happen in the browser — the photo is never uploaded anywhere.

This is a browser-side rewrite of the CampusX project *"Which Bollywood celebrity are
you?"*, which does the same thing in Python with MTCNN, VGGFace and Streamlit.

---

## Why it was rewritten instead of ported

The original stack cannot run on Vercel, and not for a reason that configuration fixes:

| Original | Problem on Vercel |
| --- | --- |
| Streamlit | A long-running server process. Vercel has no runtime for it. |
| TensorFlow + `keras-vggface` + MTCNN | Several hundred MB against a 250 MB serverless function limit. |
| ~8,600 celebrity images loaded at startup | Nothing persistent to load them from, and far too slow per request. |

So the pipeline was moved into the browser, where there is no function size limit and no
per-request compute cost at all. The whole deployment is static.

| | Original | This version |
| --- | --- | --- |
| Face detection | MTCNN | SSD MobileNet v1 |
| Embedding | VGGFace / ResNet50, 2048-d | ResNet (dlib port), 128-d |
| Runs on | Python server | Browser, TensorFlow.js (WebGL, WASM fallback) |
| Ranking | Cosine similarity | Euclidean distance — see below |
| Reference data | ~8,600 images on disk | One JSON of mean embeddings, ~1 KB per person |

### The metric changed, deliberately

The original ranks VGGFace embeddings by cosine similarity. These are not VGGFace
embeddings, and cosine behaves badly on them: the 128-d descriptors carry a large common
mean component, so two completely unrelated faces still score around 0.9.

Measured across six unrelated faces in this repo's test run:

```
cosine      0.835 – 0.916   (spread 0.08)
euclidean   0.598 – 0.854   (spread 0.26)
```

Ranking on cosine would report every celebrity as a ~90% match, with the ordering resting
on the third decimal place. Euclidean distance has over three times the dynamic range, and
it is the metric these descriptors were trained for — dlib's well-known 0.6 "same person"
threshold refers to it. Cosine is still computed and displayed as a secondary readout.

The percentage shown is anchored on real thresholds rather than an arbitrary rescale:
0.35 distance reads as 100, the 0.6 same-person line lands around 62, and 1.0 bottoms out
at 0.

---

## Running it

```bash
npm install
npm run dev
```

Open http://localhost:3000. It works immediately, but with no reference set loaded it will
detect your face and stop there. Load a reference set to get actual matches.

## Loading a reference set

**No celebrity photos ship with this repo.** They are other people's copyright and
likeness, and a public repo full of scraped star photos is not something to hand around.
You point the script at a dataset you have, and it writes out nothing but numbers.

1. **Get a dataset.** The Kaggle *Bollywood Celebrity Faces* dataset is the one the
   original project uses. Unzip it so you have one folder per person:

   ```
   datasets/bollywood_celeb_faces/
     Shah_Rukh_Khan/
       0001.jpg
       0002.jpg
     Alia_Bhatt/
       ...
   ```

   `datasets/` is gitignored.

2. **Install what the script needs**, once:

   ```bash
   npm run embeddings:setup
   ```

3. **Generate the embeddings:**

   ```bash
   npm run embeddings -- --input ./datasets/bollywood_celeb_faces
   ```

   This writes `public/celebrities.json` — one 128-number mean embedding per person. For
   100 people that is roughly 250 KB, and it is safe to commit: it contains no images.

4. Restart or redeploy. The page picks it up automatically.

### Script options

| Flag | Default | What it does |
| --- | --- | --- |
| `--input <dir>` | *required* | Dataset root, one folder per person |
| `--out <file>` | `public/celebrities.json` | Where to write |
| `--thumbs <dir>` | off | Also save one 256×256 thumbnail per person |
| `--max-per-person N` | `40` | Cap images per person, for speed |
| `--min-faces N` | `2` | Drop people with fewer usable faces |
| `--backend <name>` | auto | Force `wasm` instead of trying native first |
| `--crops` | off | Images are tight pre-cropped faces — upscale and pad them before detection |
| `--resume` | off | Keep people already in `--out` and continue with the rest |

The output is rewritten after every person, so a run that is interrupted (a laptop going
to sleep, a closed terminal) keeps what it finished. Rerun with `--resume` to carry on.

**The Kaggle dataset needs `--crops`.** Its images are 64×64 crops where the face fills
the frame, and SSD MobileNet misses most of them as-is (23 of 79 in one folder). Enlarging
each crop and padding it with a neutral border recovers essentially all of them (79 of 79):

```bash
npm run embeddings -- --input ./datasets/bollywood_celeb_faces --crops --max-per-person 1000
```

`--thumbs public/celebs` will cut a representative face per person if you want the app to
show pictures. That directory is gitignored, and whether you have the right to publish
those images is your call to make — the app renders fine without them.

### Backends

The script tries `@tensorflow/tfjs-node` first (native, several times faster) and falls
back to the WebAssembly backend, which installs anywhere. `embeddings:setup` installs only
the portable one, because the native package's install step downloads a platform binary
and fails on locked-down networks. For a large dataset it is worth adding:

```bash
npm install --no-save sharp @tensorflow/tfjs@4.22.0 @tensorflow/tfjs-node
```

List all three in one command: a later `--no-save` install prunes packages an earlier one
added. `@tensorflow/tfjs-node` 4.x also calls `util.isNullOrUndefined`, which Node 23
removed, so on Node 23+ it fails mid-run — pass `--backend wasm` there.

---

## Deploying

Any static-friendly host works; the app has no server component.

```bash
npm run build
```

On Vercel, import the repo and accept the detected Next.js defaults. Nothing needs
configuring.

### Where the weights come from

`public/models/` (11.9 MB) and `public/tfjs/` (1.1 MB) are **not** in git. They are copied
out of installed packages by `scripts/copy-assets.mjs`, which runs automatically before
`next build` and `next dev`:

```
@vladmandic/face-api/model/*              -> public/models/
@tensorflow/tfjs-backend-wasm/dist/*.wasm -> public/tfjs/
```

Nothing is downloaded — both packages are already dependencies — so the app still serves
every asset from its own origin with no CDN that can disappear, while the repo stays source
only. Both directories get a one-year immutable cache header, so a visitor downloads them
once. Run `npm run assets` to refresh them by hand.

### Serving the weights from a CDN instead

Set `NEXT_PUBLIC_ASSET_CDN` to a package CDN root to skip the local copies and fetch the
same files from there, which trims ~13 MB off the deployment:

```
NEXT_PUBLIC_ASSET_CDN=https://cdn.jsdelivr.net/npm
```

Leave it empty or unset for the self-hosted default.

---

## How it fits together

```
app/
  layout.tsx          fonts, metadata
  page.tsx            renders the client app
  globals.css         the whole stylesheet
  icon.svg            favicon
components/
  MatchApp.tsx        upload, detection, results
lib/
  faceapi.ts          model loading, backend selection, face description
  match.ts            cosine, euclidean, percentage mapping, ranking
scripts/
  copy-assets.mjs       weights from node_modules -> public/ (runs on prebuild)
  build-embeddings.mjs  dataset -> public/celebrities.json
public/
  celebrities.json    the reference set (committed; numbers only)
  models/             face-api weights, generated, gitignored
  tfjs/               TensorFlow.js WASM binaries, generated, gitignored
```

The first photo triggers a ~12 MB model download, which the browser then caches. Matching
after that is immediate.

---

## A caveat worth keeping

A lookalike score is a toy. The number says how close two face embeddings sit in a vector
space, which is not the same thing as looking alike to a person, and it is certainly not an
identity check. Face recognition also carries well-documented accuracy differences across
skin tone, age and gender, and a model trained largely on Western faces will not be even-
handed here. Please don't use this for anything that matters.

## Licence

MIT for the code in this repository. The models carry their own licences from the face-api
project, and any dataset you supply carries its own terms.

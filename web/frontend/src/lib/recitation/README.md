# Recitation engine (clean-room)

MIT-licensed TypeScript reimplementation of the offline Quran recitation
pipeline specified in `docs/specs/recitation-engine-spec.md`.

Provenance: algorithms were written from that behavioural spec plus
`docs/specs/vectors/` oracles. alketab's published engine is the design
source and remains research-only under `experiments/prompter-zipformer/engine/`;
this package does not contain that source.

Host integration: `src/worker/zipformer-session.ts` drives Kaldi fbank →
streaming Zipformer2-CTC → this engine → the 50% ayah gate in
`src/lib/zipformer-emission.ts`.

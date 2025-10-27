# Protocol Buffer Basics: Python (Reference)

Source: [https://protobuf.dev/getting-started/pythontutorial/](https://protobuf.dev/getting-started/pythontutorial/)

Key points:
- Walks through defining messages in `.proto` files, compiling them with `protoc`, and working with the generated Python API.
- Highlights that the Python package layout, not the `package` directive, governs imports—generated files typically import peers via relative paths.
- Emphasizes keeping the protobuf compiler and runtime versions compatible to avoid runtime warnings or failures.
- Shows canonical usage patterns for writing, reading, and evolving protocol buffer messages.

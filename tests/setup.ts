// Global test setup
// Mock the transformers library to avoid ES module issues in Jest
jest.mock('@xenova/transformers', () => ({
  pipeline: jest.fn().mockResolvedValue(
    jest.fn().mockResolvedValue({
      data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]) // Mock embedding vector
    })
  ),
}));

// The embedding service and journal scanner log progress and errors to stderr
// (stdout is reserved for the MCP stdio protocol). Suppress that here so the
// suite output stays pristine; tests that intentionally exercise a log path
// assert against this spy explicitly via `jest.mocked(console.error)`.
beforeEach(() => {
  jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  jest.mocked(console.error).mockRestore();
});
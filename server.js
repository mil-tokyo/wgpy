const express = require("express");
const app = express();

app.use(
  express.static(".", {
    setHeaders: function setHeaders(res, path, stat) {
      // no allow cache (for debugging purpose)
      res.set("Cache-Control", "no-cache");

      // CORS
      res.header("Access-Control-Allow-Origin", "*");
      res.header("Access-Control-Allow-Methods", "GET");
      res.header("Access-Control-Allow-Headers", "Content-Type");

      // COEP (for SharedArrayBuffer)
      res.set("Cross-Origin-Embedder-Policy", "require-corp");
      res.set("Cross-Origin-Opener-Policy", "same-origin");
    },
  })
);

app.listen(8000, () => {
  console.log("open http://localhost:8000/");
});

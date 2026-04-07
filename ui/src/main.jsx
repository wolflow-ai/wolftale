import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Wolftale from "./Wolftale.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Wolftale />
  </StrictMode>
);

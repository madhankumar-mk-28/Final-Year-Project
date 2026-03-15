import React from "react";

const BARS_SPINNER_CSS = `
@keyframes barsSpinnerFade {
  0%   { opacity: 1; }
  100% { opacity: 0.15; }
}
.bars-spinner-wrapper {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: var(--bars-spinner-size, 20px);
  width:  var(--bars-spinner-size, 20px);
}
.bars-spinner-inner {
  position: relative;
  top: 50%;
  left: 50%;
  height: var(--bars-spinner-size, 20px);
  width:  var(--bars-spinner-size, 20px);
}
.bars-spinner-bar {
  animation: barsSpinnerFade 1.2s linear infinite;
  background: var(--bars-spinner-color, currentColor);
  border-radius: 6px;
  height: 8%;
  left: -10%;
  position: absolute;
  top: -3.9%;
  width: 24%;
}
.bars-spinner-bar:nth-child(1)  { animation-delay: -1.2s; transform: rotate(0.0001deg) translate(146%); }
.bars-spinner-bar:nth-child(2)  { animation-delay: -1.1s; transform: rotate(30deg)     translate(146%); }
.bars-spinner-bar:nth-child(3)  { animation-delay: -1.0s; transform: rotate(60deg)     translate(146%); }
.bars-spinner-bar:nth-child(4)  { animation-delay: -0.9s; transform: rotate(90deg)     translate(146%); }
.bars-spinner-bar:nth-child(5)  { animation-delay: -0.8s; transform: rotate(120deg)    translate(146%); }
.bars-spinner-bar:nth-child(6)  { animation-delay: -0.7s; transform: rotate(150deg)    translate(146%); }
.bars-spinner-bar:nth-child(7)  { animation-delay: -0.6s; transform: rotate(180deg)    translate(146%); }
.bars-spinner-bar:nth-child(8)  { animation-delay: -0.5s; transform: rotate(210deg)    translate(146%); }
.bars-spinner-bar:nth-child(9)  { animation-delay: -0.4s; transform: rotate(240deg)    translate(146%); }
.bars-spinner-bar:nth-child(10) { animation-delay: -0.3s; transform: rotate(270deg)    translate(146%); }
.bars-spinner-bar:nth-child(11) { animation-delay: -0.2s; transform: rotate(300deg)    translate(146%); }
.bars-spinner-bar:nth-child(12) { animation-delay: -0.1s; transform: rotate(330deg)    translate(146%); }
`;

let _stylesInjected = false;
function injectStyles() {
  if (_stylesInjected || document.querySelector("[data-bars-spinner]")) return;
  const style = document.createElement("style");
  style.setAttribute("data-bars-spinner", "1");
  style.textContent = BARS_SPINNER_CSS;
  document.head.appendChild(style);
  _stylesInjected = true;
}

const bars = Array(12).fill(0);

export const BarsSpinner = React.forwardRef(
  ({ className, size = 20, color = "currentColor", style: externalStyle, ...props }, ref) => {
    injectStyles();
    return (
      <div
        ref={ref}
        className={["bars-spinner-wrapper", className].filter(Boolean).join(" ")}
        style={{
          "--bars-spinner-size": `${size}px`,
          "--bars-spinner-color": color,
          ...externalStyle,
        }}
        {...props}
      >
        <div className="bars-spinner-inner">
          {bars.map((_, i) => (
            <div className="bars-spinner-bar" key={`bars-spinner-bar-${i}`} />
          ))}
        </div>
      </div>
    );
  }
);

BarsSpinner.displayName = "BarsSpinner";

export default BarsSpinner;

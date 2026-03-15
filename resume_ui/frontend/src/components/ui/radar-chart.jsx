/**
 * radar-chart.jsx
 *
 * Shadcn-style ChartContainer / ChartTooltip / ChartTooltipContent
 * adapted for this project's stack:
 *   - React 19 + JSX (no TypeScript)
 *   - Inline CSS / theme objects  (no Tailwind)
 *   - Create React App  (no "use client" directive needed)
 *
 * Drop-in replacement for the bare <ResponsiveContainer> radar chart section.
 * Colors are driven by ChartConfig, which maps config keys → CSS custom
 * properties (--color-{key}) injected into the container element.
 */

import React from "react";
import * as RechartsPrimitive from "recharts";

// ── Theme selectors (kept for future dark/light-mode CSS injection) ──────────
const THEMES = { light: "", dark: ".dark" };

// ── Context ──────────────────────────────────────────────────────────────────
const ChartContext = React.createContext(null);

function useChart() {
  const context = React.useContext(ChartContext);
  if (!context) {
    throw new Error("useChart must be used within a <ChartContainer />");
  }
  return context;
}

// ── CSS custom-property injector ─────────────────────────────────────────────
function ChartStyle({ id, config }) {
  const colorConfig = Object.entries(config).filter(
    ([, c]) => c.theme || c.color
  );
  if (!colorConfig.length) return null;

  const css = Object.entries(THEMES)
    .map(([theme, prefix]) => {
      const vars = colorConfig
        .map(([key, itemConfig]) => {
          const color =
            itemConfig.theme?.[theme] ?? itemConfig.color;
          return color ? `  --color-${key}: ${color};` : null;
        })
        .filter(Boolean)
        .join("\n");
      return `${prefix} [data-chart="${id}"] {\n${vars}\n}`;
    })
    .join("\n");

  return <style dangerouslySetInnerHTML={{ __html: css }} />;
}

// ── ChartContainer ────────────────────────────────────────────────────────────
/**
 * Wraps a Recharts chart with:
 *  1. ChartContext so ChartTooltipContent can read the config
 *  2. ChartStyle that injects --color-{key} CSS vars onto the container
 *  3. A ResponsiveContainer so charts don't need explicit width/height props
 *
 * Props
 *  config  – ChartConfig object { [key]: { label, color } }
 *  style   – inline style; set height here (e.g. style={{ height: 240 }})
 *  id      – optional; auto-generated otherwise
 */
export function ChartContainer({ id, className, children, config, style, ...props }) {
  const uid = React.useId();
  const chartId = `chart-${id || uid.replace(/:/g, "")}`;

  return (
    <ChartContext.Provider value={{ config }}>
      <div
        data-slot="chart"
        data-chart={chartId}
        className={className}
        style={{
          display: "flex",
          justifyContent: "center",
          width: "100%",
          ...style,
        }}
        {...props}
      >
        <ChartStyle id={chartId} config={config} />
        <RechartsPrimitive.ResponsiveContainer width="100%" height="100%">
          {children}
        </RechartsPrimitive.ResponsiveContainer>
      </div>
    </ChartContext.Provider>
  );
}

// ── ChartTooltip ──────────────────────────────────────────────────────────────
// Re-export Recharts' Tooltip so callers use a unified import path.
export const ChartTooltip = RechartsPrimitive.Tooltip;

// Indicator dimensions by type
const INDICATOR_WIDTHS = { dot: 10, line: 4, dashed: 0 };
function getPayloadConfigFromPayload(config, payload, key) {
  if (typeof payload !== "object" || payload === null) return undefined;

  const payloadPayload =
    "payload" in payload &&
    typeof payload.payload === "object" &&
    payload.payload !== null
      ? payload.payload
      : undefined;

  let configLabelKey = key;

  if (key in payload && typeof payload[key] === "string") {
    configLabelKey = payload[key];
  } else if (
    payloadPayload &&
    key in payloadPayload &&
    typeof payloadPayload[key] === "string"
  ) {
    configLabelKey = payloadPayload[key];
  }

  return configLabelKey in config
    ? config[configLabelKey]
    : config[key];
}

// ── ChartTooltipContent ───────────────────────────────────────────────────────
/**
 * Styled tooltip that reads from ChartContext.
 *
 * Extra prop  themeColors  – pass the current theme object (C) from the
 *   parent component so the tooltip matches the app's active theme:
 *     content={<ChartTooltipContent themeColors={C} />}
 */
export function ChartTooltipContent({
  active,
  payload,
  label,
  labelFormatter,
  formatter,
  color,
  nameKey,
  labelKey,
  hideLabel = false,
  hideIndicator = false,
  indicator = "dot",
  // Project-specific: pass the theme object (C) to match app colours.
  themeColors = {},
}) {
  const { config } = useChart();

  // Resolve tooltip theme colours with sensible dark-mode fallbacks.
  const bg     = themeColors.drawerBg  || themeColors.bg  || "#0f0f11";
  const border = themeColors.border    || "rgba(255,255,255,0.09)";
  const text   = themeColors.text      || "#f4f4f5";
  const sub    = themeColors.sub       || "#71717a";

  const tooltipLabel = React.useMemo(() => {
    if (hideLabel || !payload?.length) return null;
    const [item] = payload;
    const key = `${labelKey || item?.dataKey || item?.name || "value"}`;
    const itemConfig = getPayloadConfigFromPayload(config, item, key);
    const value =
      !labelKey && typeof label === "string"
        ? config[label]?.label || label
        : itemConfig?.label;
    if (!value) return null;
    const content = labelFormatter
      ? labelFormatter(value, payload)
      : value;
    return (
      <div style={{ fontWeight: 600, marginBottom: 4, color: text }}>
        {content}
      </div>
    );
  }, [label, labelFormatter, payload, hideLabel, config, labelKey, text]);

  if (!active || !payload?.length) return null;

  const nestLabel = payload.length === 1 && indicator !== "dot";

  return (
    <div
      style={{
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 10,
        padding: "8px 12px",
        fontSize: 11,
        backdropFilter: "blur(10px)",
        boxShadow: "0 8px 32px rgba(0,0,0,.35)",
        color: text,
        minWidth: "8rem",
      }}
    >
      {!nestLabel && tooltipLabel}
      <div style={{ display: "grid", gap: 5 }}>
        {payload.map((item, index) => {
          const key = `${nameKey || item.name || item.dataKey || "value"}`;
          const itemConfig = getPayloadConfigFromPayload(config, item, key);
          const indicatorColor = color || item.payload?.fill || item.color;

          return (
            <div
              key={item.dataKey || index}
              style={{ display: "flex", alignItems: "center", gap: 8 }}
            >
              {formatter && item?.value !== undefined && item.name ? (
                formatter(item.value, item.name, item, index, item.payload)
              ) : (
                <>
                  {!hideIndicator && (
                    itemConfig?.icon ? (
                      <itemConfig.icon />
                    ) : (
                      <div
                        style={{
                          width: INDICATOR_WIDTHS[indicator] ?? INDICATOR_WIDTHS.dot,
                          height: 10,
                          borderRadius: 2,
                          background: indicatorColor,
                          border:
                            indicator === "dashed"
                              ? `1.5px dashed ${indicatorColor}`
                              : "none",
                          flexShrink: 0,
                        }}
                      />
                    )
                  )}
                  <div
                    style={{
                      display: "flex",
                      flex: 1,
                      justifyContent: "space-between",
                      alignItems: nestLabel ? "flex-end" : "center",
                    }}
                  >
                    <div>
                      {nestLabel && tooltipLabel}
                      <span style={{ color: sub }}>
                        {itemConfig?.label || item.name}
                      </span>
                    </div>
                    {item.value !== undefined && (
                      <span
                        style={{
                          color: text,
                          fontWeight: 700,
                          fontFamily: "monospace",
                          marginLeft: 12,
                        }}
                      >
                        {typeof item.value === "number"
                          ? item.value.toLocaleString()
                          : item.value}
                      </span>
                    )}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

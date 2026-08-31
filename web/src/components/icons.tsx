/**
 * Thin line icons matching the mockup's weight.
 *
 * Hand-drawn rather than exported: the Figma draft is a flattened PNG with no vector
 * layers, so there was nothing to export. Each glyph is a 24x24 stroked path at
 * 1.6 weight, which is what the render shows.
 */
import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Icon({ size = 18, children, ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

export const SparkIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3v6M12 15v6M3 12h6M15 12h6" />
    <path d="M7.5 7.5 10 10M14 14l2.5 2.5M16.5 7.5 14 10M10 14l-2.5 2.5" />
  </Icon>
)

export const PanelIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2.5" />
    <path d="M10 4v16" />
  </Icon>
)

export const PlusIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
)

export const SearchIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4 4" />
  </Icon>
)

export const GlobeIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M3.5 12h17M12 3.5c2.2 2.4 3.3 5.4 3.3 8.5S14.2 18.1 12 20.5c-2.2-2.4-3.3-5.4-3.3-8.5S9.8 5.9 12 3.5Z" />
  </Icon>
)

export const BookIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 4.5h9a3 3 0 0 1 3 3v12a2.5 2.5 0 0 0-2.5-2.5H5Z" />
    <path d="M5 4.5v12.5" />
    <path d="M19.5 6.5v13" />
  </Icon>
)

export const TrayIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 14.5h4l1.5 2.5h6l1.5-2.5h4" />
    <path d="M6 4.5h12l2.5 10v3a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2v-3Z" />
  </Icon>
)

export const HistoryIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1" />
    <path d="M3.5 4.5v4h4" />
    <path d="M12 8v4.5l3 1.8" />
  </Icon>
)

export const LogoutIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M14 4.5h3.5a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H14" />
    <path d="M10 8.5 6.5 12 10 15.5M6.5 12H15" />
  </Icon>
)

export const ChevronDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m6 9.5 6 5 6-5" />
  </Icon>
)

export const DotsIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="5.5" cy="12" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="18.5" cy="12" r="1.2" fill="currentColor" stroke="none" />
  </Icon>
)

export const LinkIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10 13.5a3.5 3.5 0 0 0 5 0l3-3a3.5 3.5 0 0 0-5-5l-1 1" />
    <path d="M14 10.5a3.5 3.5 0 0 0-5 0l-3 3a3.5 3.5 0 0 0 5 5l1-1" />
  </Icon>
)

export const DownloadIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 4v10M8 10.5l4 4 4-4" />
    <path d="M4.5 17.5v1a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-1" />
  </Icon>
)

export const ClipIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M17 8.5 10 15.5a2.5 2.5 0 0 1-3.5-3.5l7.5-7.5a4 4 0 0 1 5.5 5.5l-8 8a5.5 5.5 0 0 1-8-7.5" />
  </Icon>
)

export const BulbIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M9.5 17.5h5M10.5 20.5h3" />
    <path d="M12 3.5a5.5 5.5 0 0 0-3.2 10c.5.4.7.9.7 1.5h5c0-.6.2-1.1.7-1.5A5.5 5.5 0 0 0 12 3.5Z" />
  </Icon>
)

export const LayersIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m12 3.5 8.5 4.5-8.5 4.5L3.5 8Z" />
    <path d="m3.5 12.5 8.5 4.5 8.5-4.5" />
  </Icon>
)

export const SendIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </Icon>
)

export const TranslateIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 6h8M7.5 4v2M9.5 6c-.5 4-3 6.5-6 8" />
    <path d="M5.5 10.5c1.5 2 3.5 3 5.5 3.5" />
    <path d="m13 20 3.5-9 3.5 9M14.4 17h5.2" />
  </Icon>
)

export const HelpIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M9.8 9.6a2.2 2.2 0 1 1 2.9 2.1c-.6.2-.9.7-.9 1.3v.4" />
    <circle cx="12" cy="16.6" r="0.8" fill="currentColor" stroke="none" />
  </Icon>
)

export const TrashIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4.5 6.5h15M9.5 6.5V5a1.5 1.5 0 0 1 1.5-1.5h2A1.5 1.5 0 0 1 14.5 5v1.5" />
    <path d="M6.5 6.5 7.4 19a1.5 1.5 0 0 0 1.5 1.4h6.2a1.5 1.5 0 0 0 1.5-1.4l.9-12.5" />
  </Icon>
)

export const RefreshIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M20 12a8 8 0 1 1-2.3-5.6" />
    <path d="M20 4v4.5h-4.5" />
  </Icon>
)

export const FileIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M13.5 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9Z" />
    <path d="M13.5 3.5V9H19" />
  </Icon>
)

export const ChartIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.5v8.5h8.5A8.5 8.5 0 1 0 12 3.5Z" />
    <path d="M12 12 18 18" />
  </Icon>
)

export const FeatherIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M19.5 4.5c-6 0-11 3.5-11 9v3l-4 3" />
    <path d="M8.5 13.5h6M19.5 4.5c0 6-3.5 11-9 11" />
  </Icon>
)

export const CloseIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m6 6 12 12M18 6 6 18" />
  </Icon>
)

export const SlidersIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 7h14M5 12h14M5 17h14" />
    <circle cx="9" cy="7" r="2" fill="white" />
    <circle cx="15" cy="12" r="2" fill="white" />
    <circle cx="10" cy="17" r="2" fill="white" />
  </Icon>
)

export const GraphIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="6" cy="7" r="2.5" />
    <circle cx="18" cy="6.5" r="2.5" />
    <circle cx="12" cy="17.5" r="2.5" />
    <path d="M8.2 8.4 10.6 15M15.9 8.3 13.6 15.4M8.4 6.6h7.1" />
  </Icon>
)

/** The gradient orb from the empty state. */
export function Orb({ size = 116 }: { size?: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="rounded-full bg-[radial-gradient(circle_at_32%_28%,#FFFFFF_0%,#E8DFFC_28%,#CCB3FC_62%,#A78BFA_100%)]
        shadow-[0_18px_45px_-16px_rgba(139,92,246,0.55)]"
      aria-hidden="true"
    />
  )
}

/** Square app mark used in the sidebar and the model pill. */
export function BrandMark({ size = 26 }: { size?: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="flex items-center justify-center rounded-[8px]
        bg-[linear-gradient(140deg,#CCB3FC_0%,#8B5CF6_100%)] text-white"
      aria-hidden="true"
    >
      <SparkIcon size={Math.round(size * 0.62)} strokeWidth={2} />
    </div>
  )
}

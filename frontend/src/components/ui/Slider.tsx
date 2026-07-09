import './Slider.css'

interface SliderProps {
  id: string
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  displayValue?: (value: number) => string
}

export function Slider({
  id,
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  displayValue,
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className="slider">
      <div className="slider__header">
        <label htmlFor={id} className="slider__label">
          {label}
        </label>
        <span className="slider__value">
          {displayValue ? displayValue(value) : value}
        </span>
      </div>
      <input
        type="range"
        id={id}
        className="slider__input"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          background: `linear-gradient(to right, var(--isu-cardinal) ${percentage}%, var(--isu-border) ${percentage}%)`,
        }}
      />
    </div>
  )
}

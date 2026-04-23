import './Checkbox.css'

interface CheckboxProps {
  id: string
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
}

export function Checkbox({
  id,
  label,
  checked,
  onChange,
  disabled = false,
}: CheckboxProps) {
  return (
    <label className={`checkbox ${disabled ? 'checkbox--disabled' : ''}`} htmlFor={id}>
      <input
        type="checkbox"
        id={id}
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="checkbox__input"
      />
      <span className="checkbox__box">
        <svg
          className="checkbox__check"
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="2 6 5 9 10 3" />
        </svg>
      </span>
      <span className="checkbox__label">{label}</span>
    </label>
  )
}

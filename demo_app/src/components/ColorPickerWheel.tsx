import React from 'react';

const COLORS = ['#FF0000', '#007BFF', '#000000', '#FFFFFF', '#FFD700', '#00FF7F', '#800080'];

type Props = {
  selected: string;
  onChange: (color: string) => void;
};

export default function ColorPickerWheel({ selected, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {COLORS.map((hex) => (
        <button
          key={hex}
          className={`w-8 h-8 rounded-full border-2 ${
            selected === hex ? 'border-white' : 'border-zinc-600'
          }`}
          style={{ backgroundColor: hex }}
          onClick={() => onChange(hex)}
        />
      ))}
    </div>
  );
}

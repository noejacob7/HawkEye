import React from 'react';

type Props = {
  label: string;
  value: number; // from 0.0 to 1.0
};

export default function StatBar({ label, value }: Props) {
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <div className="w-full bg-zinc-700 rounded h-3 overflow-hidden">
        <div
          className="bg-blue-500 h-full rounded transition-all duration-300"
          style={{ width: `${value * 100}%` }}
        ></div>
      </div>
    </div>
  );
}

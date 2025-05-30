import React from 'react';
import { CarInfo } from '../data/cars';

type Props = {
  car: CarInfo;
  selected: boolean;
  onClick: () => void;
};

export default function CarCard({ car, selected, onClick }: Props) {
  const { name, image, stats } = car;

  return (
    <div
      onClick={onClick}
      className={`bg-zinc-800 p-4 rounded-lg cursor-pointer transition-all border-2 ${
        selected ? 'border-blue-500 scale-[1.03]' : 'border-transparent'
      }`}
    >
      <img src={image} alt={name} className="w-full h-32 object-contain mb-2" />
      <h3 className="text-lg font-bold mb-1">{name}</h3>
      <div className="text-sm text-gray-400 space-y-1">
        <div>Top Speed: {stats.topSpeed} km/h</div>
        <div>Acceleration: {stats.acceleration}s</div>
        <div>Grip: {stats.grip}</div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { CarInfo } from '../data/cars';
import CarSelector3D from '../components/CarSelector3D';
import ColorPickerWheel from '../components/ColorPickerWheel';
import StatBar from '../components/StatBar';

export default function CustomizeCarScene() {
  const { state } = useLocation();
  const car = state as CarInfo;
  const [color, setColor] = useState('#FF0000');
  const navigate = useNavigate();

  if (!car) return <div className="text-white p-6">No car selected.</div>;

  return (
    <div className="min-h-screen bg-zinc-900 text-white p-6">
      <h1 className="text-3xl font-bold mb-2">{car.name}</h1>
      <p className="text-sm text-gray-400 mb-4 capitalize">{car.category}</p>

      <div className="w-full h-[400px] mb-4">
        <CarSelector3D modelPath={car.modelPath} color={color} />
      </div>

      <div className="mb-4">
        <ColorPickerWheel selected={color} onChange={setColor} />
      </div>

      <div className="space-y-2 mb-6">
        <StatBar label="Top Speed" value={car.stats.topSpeed / 400} />
        <StatBar label="Acceleration" value={1 - car.stats.acceleration / 10} />
        <StatBar label="Grip" value={car.stats.grip / 100} />
        <StatBar label="Fuel" value={car.stats.fuel / 100} />
        <StatBar label="Nitro" value={car.stats.nitro / 100} />
      </div>

      <div className="flex space-x-4">
        <button
          onClick={() => navigate('/')}
          className="bg-gray-600 px-5 py-2 rounded hover:bg-gray-500"
        >
          Back
        </button>
        <button
          onClick={() =>
            navigate('/radar', {
              state: { modelPath: car.modelPath, color, name: car.name },
            })
          }
          className="bg-green-600 px-5 py-2 rounded hover:bg-green-700"
        >
          Confirm and Start
        </button>
      </div>
    </div>
  );
}

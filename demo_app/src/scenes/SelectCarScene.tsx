// src/scenes/SelectCarScene.tsx
import React, { useState } from 'react';
import CarSelector3D from '../components/CarSelector3D';
import { useNavigate } from 'react-router-dom';

const carList = [
  { name: 'CAR Model', file: '/cars/CAR Model.glb' },
  { name: 'Car', file: '/cars/Car.glb' },
  { name: 'Range Rover', file: '/cars/Range Rover.glb' },
];

const colorOptions = ['#FF0000', '#007BFF', '#000000', '#FFFFFF', '#FFD700'];

export default function SelectCarScene() {
  const [selectedCar, setSelectedCar] = useState(carList[0]);
  const [color, setColor] = useState('#FF0000');
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white p-4">
      <h1 className="text-3xl font-bold mb-4">Select Your Car</h1>
      <div className="flex items-center space-x-4 mb-4">
        {carList.map((car) => (
          <button
            key={car.name}
            onClick={() => setSelectedCar(car)}
            className={`px-4 py-2 rounded ${
              selectedCar.name === car.name ? 'bg-blue-600' : 'bg-gray-700'
            }`}
          >
            {car.name}
          </button>
        ))}
      </div>

      <div className="w-full h-[400px] mb-4">
        <CarSelector3D modelPath={selectedCar.file} color={color} />
      </div>

      <div className="flex space-x-2 mb-4">
        {colorOptions.map((hex) => (
          <button
            key={hex}
            className="w-8 h-8 rounded-full border-2 border-white"
            style={{ backgroundColor: hex }}
            onClick={() => setColor(hex)}
          />
        ))}
      </div>

      <button
        onClick={() => navigate('/radar', { state: { model: selectedCar, color } })}
        className="bg-green-600 px-6 py-2 rounded hover:bg-green-700"
      >
        Confirm Selection
      </button>
    </div>
  );
}

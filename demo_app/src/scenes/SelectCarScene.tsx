import React, { useState } from 'react';
import CategoryTabs from '../components/CategoryTabs';
import CarGrid from '../components/CarGrid';
import { carData, CarCategory, CarInfo } from '../data/cars';
import { useNavigate } from 'react-router-dom';

export default function SelectCarScreen() {
  const [category, setCategory] = useState<CarCategory>('Sport');
  const [selectedCar, setSelectedCar] = useState<CarInfo | null>(null);
  const navigate = useNavigate();

  const filteredCars = carData.filter((car) => car.category === category);

  return (
    <div className="p-6 text-white bg-zinc-900 min-h-screen">
      <h1 className="text-3xl font-bold mb-4">Choose Your Car</h1>
      <CategoryTabs selected={category} onSelect={setCategory} />
      <CarGrid cars={filteredCars} selectedCar={selectedCar} onSelectCar={setSelectedCar} />
      <div className="mt-6">
        <button
          disabled={!selectedCar}
          onClick={() => navigate('/customize', { state: selectedCar })}
          className={`px-6 py-3 text-lg rounded ${
            selectedCar ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 cursor-not-allowed'
          }`}
        >
          {selectedCar ? 'Customize Car' : 'Select a Car'}
        </button>
      </div>
    </div>
  );
}

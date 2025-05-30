import React from 'react';
import CarCard from './CarCard';
import { CarInfo } from '../data/cars';

type Props = {
  cars: CarInfo[];
  selectedCar: CarInfo | null;
  onSelectCar: (car: CarInfo) => void;
};

export default function CarGrid({ cars, selectedCar, onSelectCar }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {cars.map((car) => (
        <CarCard
          key={car.id}
          car={car}
          selected={selectedCar?.id === car.id}
          onClick={() => onSelectCar(car)}
        />
      ))}
    </div>
  );
}

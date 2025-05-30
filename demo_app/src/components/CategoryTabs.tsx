import React from 'react';
import type { CarCategory } from '../data/cars';

type Props = {
  selected: CarCategory;
  onSelect: (category: CarCategory) => void;
};

const categories: CarCategory[] = ['Sport', 'SUV', 'Classic'];

export default function CategoryTabs({ selected, onSelect }: Props) {
  return (
    <div className="flex space-x-4 mb-6">
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => onSelect(cat)}
          className={`px-4 py-2 rounded-full text-sm font-semibold uppercase transition-all duration-200 ${
            selected === cat ? 'bg-blue-600 text-white' : 'bg-zinc-700 text-gray-300 hover:bg-zinc-600'
          }`}
        >
          {cat}
        </button>
      ))}
    </div>
  );
}

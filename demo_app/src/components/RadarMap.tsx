import React from "react";
import { Stage, Layer, Rect, Image as KonvaImage } from "react-konva";

const RadarMap: React.FC = () => {
  const width = 800;
  const height = 600;

  return (
    <div className="w-full h-full flex justify-center items-center">
      <Stage width={width} height={height} className="bg-gray-100 border border-gray-400">
        <Layer>
          {/* Background grid */}
          {[...Array(20)].map((_, i) => (
            <Rect
              key={"v-" + i}
              x={(width / 20) * i}
              y={0}
              width={1}
              height={height}
              fill="#ddd"
            />
          ))}
          {[...Array(15)].map((_, i) => (
            <Rect
              key={"h-" + i}
              x={0}
              y={(height / 15) * i}
              width={width}
              height={1}
              fill="#ddd"
            />
          ))}

          {/* Example UAV and car placement (for now static) */}
          <Rect x={100} y={100} width={40} height={40} fill="blue" cornerRadius={5} />
          <Rect x={300} y={200} width={60} height={30} fill="red" cornerRadius={4} />
        </Layer>
      </Stage>
    </div>
  );
};

export default RadarMap;

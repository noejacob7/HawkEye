// src/components/CarSelector3D.tsx
import React, { useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

type Props = {
  modelPath: string;
  color: string;
};

const CarModel = ({ modelPath, color }: Props) => {
  const gltf = useGLTF(modelPath);

  useEffect(() => {
    gltf.scene.traverse((child: any) => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({ color });
      }
    });
  }, [color, gltf]);

  return <primitive object={gltf.scene} scale={2.5} />;
};

export default function CarSelector3D({ modelPath, color }: Props) {
  return (
    <Canvas camera={{ position: [0, 1.5, 4] }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[10, 10, 5]} />
      <CarModel modelPath={modelPath} color={color} />
      <OrbitControls />
      <Environment preset="city" />
    </Canvas>
  );
}

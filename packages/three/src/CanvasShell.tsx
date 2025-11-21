import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';

interface CanvasShellProps {
    children: React.ReactNode;
}

export const CanvasShell: React.FC<CanvasShellProps> = ({ children }) => {
    return (
        <div style={{ width: '100%', height: '100%', minHeight: '500px' }}>
            <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                <Suspense fallback={null}>
                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} />
                    {children}
                    <OrbitControls makeDefault />
                    <Environment preset="city" />
                </Suspense>
            </Canvas>
        </div>
    );
};

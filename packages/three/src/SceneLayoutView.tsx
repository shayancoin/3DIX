import React from 'react';
import { Box, Grid } from '@react-three/drei';

export const SceneLayoutView: React.FC = () => {
    return (
        <group>
            <Grid infiniteGrid fadeDistance={50} fadeStrength={5} />
            <Box position={[0, 0.5, 0]}>
                <meshStandardMaterial color="hotpink" />
            </Box>
        </group>
    );
};

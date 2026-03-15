import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * DottedSurface – animated Three.js particle wave background.
 * Props:
 *   dark  {boolean} – true = dark theme (light dots), false = light theme (dark dots)
 *   style {object}  – additional inline styles for the container div
 */
export function DottedSurface({ dark = true, style = {} }) {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);

    useEffect(() => {
        if (!containerRef.current) return;
        const container = containerRef.current;

        const SEPARATION = 150;
        const AMOUNTX = 40;
        const AMOUNTY = 60;

        // Scene setup
        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0xffffff, 2000, 10000);

        const camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            1,
            10000
        );
        camera.position.set(0, 355, 1220);

        const renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true,
        });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(scene.fog.color, 0);

        container.appendChild(renderer.domElement);

        // Build geometry
        const positions = [];
        const colors = [];
        const geometry = new THREE.BufferGeometry();

        // Dot color: light grey in dark mode, near-black in light mode
        const [r, g, b] = dark ? [0.78, 0.78, 0.78] : [0.12, 0.12, 0.12];

        for (let ix = 0; ix < AMOUNTX; ix++) {
            for (let iy = 0; iy < AMOUNTY; iy++) {
                positions.push(
                    ix * SEPARATION - (AMOUNTX * SEPARATION) / 2,
                    0,
                    iy * SEPARATION - (AMOUNTY * SEPARATION) / 2
                );
                colors.push(r, g, b);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 8,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true,
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        let count = 0;
        let animationId;

        const animate = () => {
            animationId = requestAnimationFrame(animate);

            const posAttr = geometry.attributes.position;
            const arr = posAttr.array;

            let i = 0;
            for (let ix = 0; ix < AMOUNTX; ix++) {
                for (let iy = 0; iy < AMOUNTY; iy++) {
                    arr[i * 3 + 1] =
                        Math.sin((ix + count) * 0.3) * 50 +
                        Math.sin((iy + count) * 0.5) * 50;
                    i++;
                }
            }
            posAttr.needsUpdate = true;

            renderer.render(scene, camera);
            count += 0.1;
        };

        const handleResize = () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        };

        window.addEventListener('resize', handleResize);
        animate();

        sceneRef.current = { scene, camera, renderer, animationId };

        return () => {
            window.removeEventListener('resize', handleResize);
            if (sceneRef.current) {
                cancelAnimationFrame(sceneRef.current.animationId);
                sceneRef.current.scene.traverse((obj) => {
                    if (obj instanceof THREE.Points) {
                        obj.geometry.dispose();
                        if (Array.isArray(obj.material)) {
                            obj.material.forEach((m) => m.dispose());
                        } else {
                            obj.material.dispose();
                        }
                    }
                });
                sceneRef.current.renderer.dispose();
                if (container && sceneRef.current.renderer.domElement) {
                    container.removeChild(sceneRef.current.renderer.domElement);
                }
            }
        };
    }, [dark]); // re-run when theme changes

    return (
        <div
            ref={containerRef}
            style={{
                pointerEvents: 'none',
                position: 'fixed',
                inset: 0,
                zIndex: -1,
                ...style,
            }}
        />
    );
}

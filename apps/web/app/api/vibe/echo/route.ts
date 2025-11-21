import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    // Proxy to backend API if available
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/vibe/echo`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      // Fallback to local echo if backend is not available
      console.warn('Backend API not available, using local echo:', error);
    }

    // Fallback: local echo
    return NextResponse.json({ message: `Echo: ${body.message}` });
  } catch (error) {
    return NextResponse.json(
      { error: 'Invalid request' },
      { status: 400 }
    );
  }
}

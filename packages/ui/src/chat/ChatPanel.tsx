import React, { useState } from 'react';

export const ChatPanel: React.FC = () => {
    const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        if (!input.trim()) return;
        const newMessages = [...messages, { role: 'user', content: input }];
        setMessages(newMessages);
        setInput('');

        try {
            const res = await fetch('/api/vibe/echo', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input }),
            });
            const data = await res.json();
            setMessages([...newMessages, { role: 'assistant', content: data.message }]);
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="flex flex-col h-full border-l w-80 bg-white">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((m, i) => (
                    <div key={i} className={`p-2 rounded ${m.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'}`}>
                        {m.content}
                    </div>
                ))}
            </div>
            <div className="p-4 border-t">
                <input
                    className="w-full p-2 border rounded"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Describe your vibe..."
                />
            </div>
        </div>
    );
};

"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    await authClient.signIn.email(
      {
        email,
        password,
        callbackURL: "/",
      },
      {
        onSuccess: () => {
          router.push("/");
        },
        onError: (ctx) => {
          setError(ctx.error.message || "Invalid credentials");
        },
      }
    );

    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      <form
        onSubmit={handleLogin}
        className="w-full max-w-sm space-y-4 rounded border p-6"
      >
        <h1 className="text-xl font-bold">Login</h1>

        {error && <p className="text-red-500">{error}</p>}

        <input
          type="email"
          placeholder="Email"
          className="w-full border p-2"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Password"
          className="w-full border p-2"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-black p-2 text-white disabled:opacity-50"
        >
          {loading ? "Signing in..." : "Login"}
        </button>

        <p className="text-sm">
          Don’t have an account?{" "}
          <span
            className="cursor-pointer underline"
            onClick={() => router.push("/signup")}
          >
            Sign up
          </span>
        </p>
      </form>
    </div>
  );
}

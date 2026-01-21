"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

export default function SignupPage() {
  const router = useRouter();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    await authClient.signUp.email(
      {
        name,
        email,
        password,
        callbackURL: "/",
      },
      {
        onSuccess: () => {
          router.push("/");
        },
        onError: (ctx) => {
          setError(ctx.error.message || "Signup failed");
        },
      }
    );

    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      <form
        onSubmit={handleSignup}
        className="w-full max-w-sm space-y-4 rounded border p-6"
      >
        <h1 className="text-xl font-bold">Create Account</h1>

        {error && <p className="text-red-500">{error}</p>}

        <input
          type="text"
          placeholder="Name"
          className="w-full border p-2"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />

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
          placeholder="Password (min 8 chars)"
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
          {loading ? "Creating..." : "Sign Up"}
        </button>

        <p className="text-sm">
          Already have an account?{" "}
          <span
            className="cursor-pointer underline"
            onClick={() => router.push("/login")}
          >
            Login
          </span>
        </p>
      </form>
    </div>
  );
}

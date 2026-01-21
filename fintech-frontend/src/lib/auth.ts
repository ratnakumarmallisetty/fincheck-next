import { betterAuth } from "better-auth";
import { mongodbAdapter } from "better-auth/adapters/mongodb";
import { nextCookies } from "better-auth/next-js";
import { connectMongo } from "./mongodb";

const db = await connectMongo();

export const auth = betterAuth({
  database: mongodbAdapter(db),

  emailAndPassword: {
    enabled: true,
  },

  plugins: [
    nextCookies(), // MUST be last
  ],
});

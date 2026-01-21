"use client";

import BankingDemoPage from "@/app/banking-demo/page";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Slider() {
  const pathname = usePathname();

  const tabs = [
    { label: "Upload", href: "/upload" },
    { label: "Results", href: "/results" },
    { label: "Verify Image", href: "/banking-demo" }
  ];

  return (
    <div className="flex justify-center">
      <div
        role="tablist"
        className="tabs tabs-bordered bg-base-100 text-base-content"
      >
        {tabs.map((tab) => (
          <Link
            key={tab.href}
            href={tab.href}
            role="tab"
            className={`tab px-6 ${
              pathname === tab.href
                ? "tab-active font-semibold text-primary"
                : "opacity-100 hover:opacity-100"
            }`}
          >
            {tab.label}
          </Link>
        ))}
      </div>
    </div>
  );
}

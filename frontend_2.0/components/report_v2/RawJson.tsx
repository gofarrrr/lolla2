"use client";

import React from 'react'

export function RawJson({ data, title }: { data: any; title?: string }) {
  return (
    <section className="bg-[#1E1E1E] text-[#D4D4D4] rounded-2xl border border-[#404040] p-6 shadow-sm">
      {title && <h3 className="text-lg font-semibold mb-3 text-[#D4D4D4]">{title}</h3>}
      <pre className="text-[13px] leading-5 font-mono whitespace-pre-wrap">
{JSON.stringify(data ?? {}, null, 2)}
      </pre>
    </section>
  )
}

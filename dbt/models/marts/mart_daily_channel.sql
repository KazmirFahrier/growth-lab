-- One row per (date, channel): ad delivery, signups, and acquisition-channel
-- attributed billing. The grain every Phase-0 metric is computed over.

with ad as (
    select date, channel, spend, impressions, clicks
    from {{ ref('stg_ad_spend_daily') }}
),

signup_agg as (
    select
        signup_date as date,
        channel,
        count(*) as signups,
        count(*) filter (where subscribed) as paid_signups
    from {{ ref('stg_signups') }}
    group by 1, 2
),

rev_agg as (
    select
        txn_date as date,
        acquisition_channel as channel,
        count(*) as txns,
        sum(amount) as revenue,
        count(*) filter (where is_fraud) as fraud_txns
    from {{ ref('fct_transactions') }}
    group by 1, 2
),

spine as (
    select date, channel from ad
    union
    select date, channel from signup_agg
    union
    select date, channel from rev_agg
)

select
    s.date,
    s.channel,
    coalesce(ad.spend, 0.0) as spend,
    coalesce(ad.impressions, 0) as impressions,
    coalesce(ad.clicks, 0) as clicks,
    coalesce(sg.signups, 0) as signups,
    coalesce(sg.paid_signups, 0) as paid_signups,
    coalesce(rv.txns, 0) as txns,
    coalesce(rv.revenue, 0.0) as revenue,
    coalesce(rv.fraud_txns, 0) as fraud_txns
from spine s
left join ad using (date, channel)
left join signup_agg sg using (date, channel)
left join rev_agg rv using (date, channel)

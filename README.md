# Pagerank Project

In this project, I created a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.
Pagerank is used to return only the most important results from this website in the search engine.

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

The graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Looking at the first ten lines, we see:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the $P$ matrix,
this is also the value of $\text{nnz}(P)$. To get the dimensions of $P$, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

## Task 1: the power method

We implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector. As mentioned above, the sparsity of matrix P greatly impacts the runtime. We use the following definition for Pagerank vector $x^k: x^k = alpha x^(k-1)^T P + (alpha x^(k-1)^T a + (1 - alpha)) v^T$.

**Part 1:**

We run the program on the `data/small.csv.gz` graph to ensure that the implementation is working. 

   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual= 0.3775096535682678
    DEBUG:root:i=1 residual= 0.15696220099925995
    DEBUG:root:i=2 residual= 0.09472890943288803
    DEBUG:root:i=3 residual= 0.04863157868385315
    DEBUG:root:i=4 residual= 0.026361854746937752
    DEBUG:root:i=5 residual= 0.015072877518832684
    DEBUG:root:i=6 residual= 0.008065990172326565
    DEBUG:root:i=7 residual= 0.004705339204519987
    DEBUG:root:i=8 residual= 0.0025796189438551664
    DEBUG:root:i=9 residual= 0.0014943481655791402
    DEBUG:root:i=10 residual= 0.0008344150264747441
    DEBUG:root:i=11 residual= 0.0004791707033291459
    DEBUG:root:i=12 residual= 0.0002701549092307687
    DEBUG:root:i=13 residual= 0.00015440073912031949
    DEBUG:root:i=14 residual= 8.740682096686214e-05
    DEBUG:root:i=15 residual= 4.979922960046679e-05
    DEBUG:root:i=16 residual= 2.825748197210487e-05
    DEBUG:root:i=17 residual= 1.6040461559896357e-05
    DEBUG:root:i=18 residual= 9.09709524421487e-06
    DEBUG:root:i=19 residual= 5.247240096650785e-06
    DEBUG:root:i=20 residual= 2.9180796445871238e-06
    DEBUG:root:i=21 residual= 1.723131049402582e-06
    DEBUG:root:i=22 residual= 9.430210070604517e-07
    INFO:root:rank=0 pagerank=7.5318e-01 url=4
    INFO:root:rank=1 pagerank=5.9124e-01 url=6
    INFO:root:rank=2 pagerank=4.6582e-01 url=5
    INFO:root:rank=3 pagerank=2.5026e-01 url=2
    INFO:root:rank=4 pagerank=1.9997e-01 url=3
    INFO:root:rank=5 pagerank=1.8171e-01 url=1
   ```

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.


**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Running the program, I get the following results:


   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
   INFO:root:rank=0 pagerank=1.4414e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 pagerank=1.2688e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 pagerank=9.5835e-04 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=3 pagerank=9.3883e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=4 pagerank=9.0297e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=5 pagerank=8.8973e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=6 pagerank=8.7221e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=7 pagerank=8.5020e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
    INFO:root:rank=8 pagerank=8.1090e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
    INFO:root:rank=9 pagerank=8.0756e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
    INFO:root:rank=0 pagerank=1.0002e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=8.2406e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=2 pagerank=8.2249e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=7.6187e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=4 pagerank=7.4243e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=5 pagerank=6.8383e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=6.4250e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=5.8655e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=5.4029e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=5.3895e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors
   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
   INFO:root:rank=0 pagerank=7.5890e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
    INFO:root:rank=1 pagerank=7.0164e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 pagerank=4.2293e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
    INFO:root:rank=3 pagerank=3.1004e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
    INFO:root:rank=4 pagerank=2.3180e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
    INFO:root:rank=5 pagerank=2.3086e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
    INFO:root:rank=6 pagerank=2.2862e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
    INFO:root:rank=7 pagerank=2.1086e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
    INFO:root:rank=8 pagerank=1.7285e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
    INFO:root:rank=9 pagerank=1.6701e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
   ```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running:

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
   INFO:root:rank=0 pagerank=5.4187e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=5.4187e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=5.4187e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=5.4187e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=5.4187e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=5.4187e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=5.4187e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=5.4187e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=5.4187e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=5.4187e-01 url=www.lawfareblog.com/topics
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
We can compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:

Task 1, Part 3
```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
   INFO:root:rank=0 pagerank=5.2268e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=4.3300e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=4.2665e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=2.0178e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=2.0027e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=5 pagerank=1.9129e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=6 pagerank=1.9046e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
    INFO:root:rank=7 pagerank=1.8587e-01 url=www.lawfareblog.com/rational-security-lonely-amigo-edition
    INFO:root:rank=8 pagerank=1.8376e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=9 pagerank=1.8376e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
   ```

We notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their $P$ matrix for the web,
they use a similar (but much more complicated) process to modify the $P$ matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<https://www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix,
and that this eigengap is bounded by the alpha parameter. We know that $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence. Changing the value of alpha gives us very different pagerank rankings. 

We run the following four commands and observe the output:

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual= 20.519126892089844
    DEBUG:root:i=1 residual= 19.25760269165039
    DEBUG:root:i=2 residual= 0.24006503820419312
    DEBUG:root:i=3 residual= 2.045966863632202
    DEBUG:root:i=4 residual= 1.1502310037612915
    DEBUG:root:i=5 residual= 0.4552375078201294
    DEBUG:root:i=6 residual= 0.4882778227329254
    DEBUG:root:i=7 residual= 0.031064631417393684
    DEBUG:root:i=8 residual= 0.24931639432907104
    DEBUG:root:i=9 residual= 0.08170860260725021
    DEBUG:root:i=10 residual= 0.08244407922029495
    DEBUG:root:i=11 residual= 0.07272566854953766
    DEBUG:root:i=12 residual= 0.009436127729713917
    DEBUG:root:i=13 residual= 0.03630061075091362
    DEBUG:root:i=14 residual= 0.010325103998184204
    DEBUG:root:i=15 residual= 0.012820490635931492
    DEBUG:root:i=16 residual= 0.009821368381381035
    DEBUG:root:i=17 residual= 0.0020453238394111395
    DEBUG:root:i=18 residual= 0.005416387226432562
    DEBUG:root:i=19 residual= 0.0014032123144716024
    DEBUG:root:i=20 residual= 0.001861126977019012
    DEBUG:root:i=21 residual= 0.0013069777050986886
    DEBUG:root:i=22 residual= 0.0005036322982050478
    DEBUG:root:i=23 residual= 0.0008919663960114121
    DEBUG:root:i=24 residual= 6.190026761032641e-05
    DEBUG:root:i=25 residual= 0.00047157175140455365
    DEBUG:root:i=26 residual= 0.00021501677110791206
    DEBUG:root:i=27 residual= 0.00011317763710394502
    DEBUG:root:i=28 residual= 0.00012758345110341907
    DEBUG:root:i=29 residual= 9.383440556121059e-06
    DEBUG:root:i=30 residual= 4.031693242723122e-05
    DEBUG:root:i=31 residual= 1.6309280908899382e-05
    DEBUG:root:i=32 residual= 5.114703981234925e-06
    DEBUG:root:i=33 residual= 4.226639703119872e-06
    DEBUG:root:i=34 residual= 2.4861424208211247e-06
    DEBUG:root:i=35 residual= 3.1120191579248058e-06
    DEBUG:root:i=36 residual= 2.5122932356680394e-07
    INFO:root:rank=0 pagerank=5.4187e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=1 pagerank=5.4187e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=5.4187e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=5.4187e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=5.4187e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=5.4187e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=5.4187e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=5.4187e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=5.4187e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=5.4187e-01 url=www.lawfareblog.com/topics

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual= 24.139978408813477
    DEBUG:root:i=1 residual= 22.823686599731445
    DEBUG:root:i=2 residual= 0.24676613509655
    DEBUG:root:i=3 residual= 0.06403284519910812
    DEBUG:root:i=4 residual= 0.020616697147488594
    DEBUG:root:i=5 residual= 0.006940558087080717
    DEBUG:root:i=6 residual= 0.002359197475016117
    DEBUG:root:i=7 residual= 0.0008085645968094468
    DEBUG:root:i=8 residual= 0.0002741555799730122
    DEBUG:root:i=9 residual= 9.573439456289634e-05
    DEBUG:root:i=10 residual= 3.2655687391525134e-05
    DEBUG:root:i=11 residual= 1.0640459549904335e-05
    DEBUG:root:i=12 residual= 4.299329702917021e-06
    DEBUG:root:i=13 residual= 1.4701284953844151e-06
    DEBUG:root:i=14 residual= 5.249833066045539e-07
    INFO:root:rank=0 pagerank=2.8866e-01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=1 pagerank=2.8866e-01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=2.8866e-01 url=www.lawfareblog.com/topics
    INFO:root:rank=3 pagerank=2.8866e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=2.8866e-01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=2.8866e-01 url=www.lawfareblog.com/masthead
    INFO:root:rank=6 pagerank=2.8866e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=7 pagerank=2.8866e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=8 pagerank=2.8866e-01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=2.8866e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
  
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual= 5.117104530334473
    DEBUG:root:i=1 residual= 4.2377166748046875
    DEBUG:root:i=2 residual= 0.38469868898391724
    DEBUG:root:i=3 residual= 0.30582183599472046
    DEBUG:root:i=4 residual= 0.0631658062338829
    DEBUG:root:i=5 residual= 0.05993955582380295
    DEBUG:root:i=6 residual= 0.01644863560795784
    DEBUG:root:i=7 residual= 0.011845304630696774
    DEBUG:root:i=8 residual= 0.003808694425970316
    DEBUG:root:i=9 residual= 0.002018704079091549
    DEBUG:root:i=10 residual= 0.0009492958779446781
    DEBUG:root:i=11 residual= 0.0004971608286723495
    DEBUG:root:i=12 residual= 0.00028877489967271686
    DEBUG:root:i=13 residual= 0.00017053441843017936
    DEBUG:root:i=14 residual= 0.00011238487059017643
    DEBUG:root:i=15 residual= 7.21962860552594e-05
    DEBUG:root:i=16 residual= 4.71628773084376e-05
    DEBUG:root:i=17 residual= 3.076593566220254e-05
    DEBUG:root:i=18 residual= 2.0043993572471663e-05
    DEBUG:root:i=19 residual= 1.2955856618646067e-05
    DEBUG:root:i=20 residual= 8.332527613674756e-06
    DEBUG:root:i=21 residual= 5.350524133973522e-06
    DEBUG:root:i=22 residual= 3.4053698527714005e-06
    DEBUG:root:i=23 residual= 2.1621074210997904e-06
    DEBUG:root:i=24 residual= 1.3623778158944333e-06
    DEBUG:root:i=25 residual= 8.536906648259901e-07
    INFO:root:rank=0 pagerank=5.2268e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=4.3300e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=4.2665e-01 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=2.0178e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=4 pagerank=2.0027e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
    INFO:root:rank=5 pagerank=1.9129e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
    INFO:root:rank=6 pagerank=1.9046e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
    INFO:root:rank=7 pagerank=1.8587e-01 url=www.lawfareblog.com/rational-security-lonely-amigo-edition
    INFO:root:rank=8 pagerank=1.8376e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
    INFO:root:rank=9 pagerank=1.8376e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
  
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual= 6.020057201385498
    DEBUG:root:i=1 residual= 5.1270952224731445
    DEBUG:root:i=2 residual= 0.4889775514602661
    DEBUG:root:i=3 residual= 0.28320351243019104
    DEBUG:root:i=4 residual= 0.18792036175727844
    DEBUG:root:i=5 residual= 0.13348814845085144
    DEBUG:root:i=6 residual= 0.1045861542224884
    DEBUG:root:i=7 residual= 0.08788826316595078
    DEBUG:root:i=8 residual= 0.07622770965099335
      ...
    DEBUG:root:i=679 residual= 1.0594972081889864e-06
    DEBUG:root:i=680 residual= 1.0491901321074693e-06
    DEBUG:root:i=681 residual= 1.0341553888792987e-06
    DEBUG:root:i=682 residual= 1.0261800298394519e-06
    DEBUG:root:i=683 residual= 1.0287457143931533e-06
    DEBUG:root:i=684 residual= 1.0042383564723423e-06
    DEBUG:root:i=685 residual= 1.0042734857051983e-06
    DEBUG:root:i=686 residual= 9.798367273106123e-07
    INFO:root:rank=0 pagerank=7.0198e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=7.0198e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.0559e-01 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 pagerank=3.1779e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 pagerank=2.2055e-02 url=www.lawfareblog.com/events
    INFO:root:rank=5 pagerank=1.6038e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 pagerank=1.6038e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 pagerank=1.6034e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 pagerank=1.6032e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
    INFO:root:rank=9 pagerank=1.6031e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   ```

We notice that the last command takes considerably more iterations (686) to compute the pagerank vector.

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best. It  is intuitive that large alpha values imply that the structure of the webgraph has more influence on the final result, and small alpha values ignore the structure of the webgraph.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
We implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.


**Part 1:**

The command line argument `--personalization_vector_query` uses the above function to augment your search with a custom personalization vector.

Implementing the function, we get the following results:
   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
   INFO:root:rank=0 pagerank=6.5252e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.5249e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.6165e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.2442e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=4 pagerank=1.2442e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=5 pagerank=9.4662e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 pagerank=9.2875e-02 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=7 pagerank=9.2875e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=8 pagerank=7.8528e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=9 pagerank=7.4127e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   ```


Notice that these results are significantly different than when using the `--search_query` option. 
Which results are better?
Again, that depends on how we define "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend a lot of time fine-tuning their Pagerank personalization vectors to remove spam webpages.
How they do this is not publically available. 

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.

   We see this implemented here:
   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
   INFO:root:rank=0 pagerank=6.5252e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.5249e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.6165e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=9.4662e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=7.1773e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=5 pagerank=7.1160e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=6 pagerank=6.5909e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=6.0301e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=5.2412e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=5.2411e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
   ```



We observe that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.



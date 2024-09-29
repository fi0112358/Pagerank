# Pagerank Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.
You will use pagerank to return only the most important results from this website in your search engine.

**Due date:** Sunday, 22 September at midnight

**Late Policy:** You lose $2^{(i-1)}$ points, where i is the number of days late.

<!--
**Computation:**
This project has low computational requirements.
You should be able to complete it on your own laptops.
-->

**Collaboration Policy:**
Do whatever will help you learn,
but be an adult.
You may talk to other students and use Google/ChatGPT.
Recall that you will have an in-person oral exam on this material and the exam is worth many more points.
The main purpose of this project is to help prepare you for the exam.

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

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.
>
> In python, we can use the built-in `gzip` module to access gzipped files.
> The following python code is equivalent to the bash code above:
>
> ```
> >>> import gzip
> >>> fin = gzip.open('data/small.csv.gz', mode='rt')
> >>> print(fin.read())
> source,target
> 1,2
> 1,3
> 3,1
> 3,2
> 3,5
> 4,5
> 4,6
> 5,6
> 5,4
> 6,4
> ```
>
> There are many terminal commands throughout these instructions.
> If you haven't used the terminal before, and so these commands are unfamiliar, that's okay.
> I'd be happy to explain them in office hours,
> or there are many tutors in the QCL available who can help.
> (There are no tutors for this class specifically, but anyone who has taken CSCI046 or CSCI133 with me will be able to help with the terminal.)
>
> Furthermore, you don't "need" to understand the terminal commands in detail,
> since you are not required to run these commands or to create your own.
> The important part is to understand the English language description of what the commands are doing,
> and to understand that this is just how I computed what the English language text is describing.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
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
this is also the value of $\text{nnz}(P)$.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of $P$, we need to count the total number of nodes in the graph.
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

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the [`FIXME: Task 1` annotation](https://github.com/mikeizbicki/cmc-csci145-math166/blob/81ed5d2b75f5bc23b8de93805c29321ab431ed9b/topic01_computation_pagerank/project/pagerank.py#L144).

> **NOTE:**
> The power method is the only data mining algorithm you will implement in class.
> You are implementing it because there are no standard library implementations available.
> Why?
> 1. The runtime is heavily dependent on the data structures used to store the graph data.
>    Different applications will need to use different data structures.
> 1. It is "trivial" to implement.
>    My solution to this homework is <10 lines of code.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to minor implementation details and weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

> **NOTE:**
> There are no automated test cases to pass for this assignment.
> Test cases for algorithms involving floating point computations are hard to write and understand.
> Minor-seeming implementations details can have large impacts on the final result.
> These software engineering issues are beyond the scope of this class.
>
> Instructions for how I will grade your homework are contained in the [submission section](#submission) at the end of this document.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

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
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.

> **NOTE:**
> It should be "obvious" to you that large alpha values imply that the structure of the webgraph has more influence on the final result,
> and small alpha values ignore the structure of the webgraph.
> Recall that the word "obvious" means that it follows directly from the definition,
> but you may still need to sit and meditate on the definition for a long period of time.

If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).
    Ensure that all of the project files are copied from this folder into your new repo.

1. As you complete the tasks above:
    Run the corresponding commands below, and paste their output into the code blocks.
    Please ensure correct markdown formatting.
   
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
    DEBUG:root:i=9 residual= 0.06707365065813065
    DEBUG:root:i=10 residual= 0.05981040000915527
    DEBUG:root:i=11 residual= 0.0544477142393589
    DEBUG:root:i=12 residual= 0.05111086368560791
    DEBUG:root:i=13 residual= 0.049810055643320084
    DEBUG:root:i=14 residual= 0.05032513663172722
    DEBUG:root:i=15 residual= 0.052217088639736176
    DEBUG:root:i=16 residual= 0.05493367463350296
    DEBUG:root:i=17 residual= 0.057919785380363464
    DEBUG:root:i=18 residual= 0.06068682298064232
    DEBUG:root:i=19 residual= 0.06284592300653458
    DEBUG:root:i=20 residual= 0.06412504613399506
    DEBUG:root:i=21 residual= 0.0643782690167427
    DEBUG:root:i=22 residual= 0.06358389556407928
    DEBUG:root:i=23 residual= 0.06182725355029106
    DEBUG:root:i=24 residual= 0.05927111208438873
    DEBUG:root:i=25 residual= 0.05611903965473175
    DEBUG:root:i=26 residual= 0.05258190631866455
    DEBUG:root:i=27 residual= 0.0488513745367527
    DEBUG:root:i=28 residual= 0.045085880905389786
    DEBUG:root:i=29 residual= 0.041404951363801956
    DEBUG:root:i=30 residual= 0.03789155185222626
    DEBUG:root:i=31 residual= 0.03459696099162102
    DEBUG:root:i=32 residual= 0.03154831752181053
    DEBUG:root:i=33 residual= 0.028754612430930138
    DEBUG:root:i=34 residual= 0.02621266059577465
    DEBUG:root:i=35 residual= 0.023911401629447937
    DEBUG:root:i=36 residual= 0.021835146471858025
    DEBUG:root:i=37 residual= 0.0199660025537014
    DEBUG:root:i=38 residual= 0.01828535459935665
    DEBUG:root:i=39 residual= 0.016774989664554596
    DEBUG:root:i=40 residual= 0.015417449176311493
    DEBUG:root:i=41 residual= 0.014196686446666718
    DEBUG:root:i=42 residual= 0.013097972609102726
    DEBUG:root:i=43 residual= 0.012107932940125465
    DEBUG:root:i=44 residual= 0.01121465116739273
    DEBUG:root:i=45 residual= 0.01040754932910204
    DEBUG:root:i=46 residual= 0.009677127934992313
    DEBUG:root:i=47 residual= 0.009015006944537163
    DEBUG:root:i=48 residual= 0.008413762785494328
    DEBUG:root:i=49 residual= 0.007866909727454185
    DEBUG:root:i=50 residual= 0.007368619088083506
    DEBUG:root:i=51 residual= 0.006913810968399048
    DEBUG:root:i=52 residual= 0.006497968919575214
    DEBUG:root:i=53 residual= 0.006117076613008976
    DEBUG:root:i=54 residual= 0.005767605733126402
    DEBUG:root:i=55 residual= 0.0054464321583509445
    DEBUG:root:i=56 residual= 0.0051507772877812386
    DEBUG:root:i=57 residual= 0.004878198262304068
    DEBUG:root:i=58 residual= 0.004626431968063116
    DEBUG:root:i=59 residual= 0.004393532872200012
    DEBUG:root:i=60 residual= 0.004177769646048546
    DEBUG:root:i=61 residual= 0.003977671265602112
    DEBUG:root:i=62 residual= 0.0037916526198387146
    DEBUG:root:i=63 residual= 0.0036185712087899446
    DEBUG:root:i=64 residual= 0.003457305021584034
    DEBUG:root:i=65 residual= 0.0033068060874938965
    DEBUG:root:i=66 residual= 0.00316616240888834
    DEBUG:root:i=67 residual= 0.0030346086714416742
    DEBUG:root:i=68 residual= 0.0029113045893609524
    DEBUG:root:i=69 residual= 0.002795690903440118
    DEBUG:root:i=70 residual= 0.0026870854198932648
    DEBUG:root:i=71 residual= 0.0025849556550383568
    DEBUG:root:i=72 residual= 0.0024888243060559034
    DEBUG:root:i=73 residual= 0.0023982012644410133
    DEBUG:root:i=74 residual= 0.0023127098102122545
    DEBUG:root:i=75 residual= 0.002231970662251115
    DEBUG:root:i=76 residual= 0.002155627589672804
    DEBUG:root:i=77 residual= 0.0020833348389714956
    DEBUG:root:i=78 residual= 0.0020148695912212133
    DEBUG:root:i=79 residual= 0.0019499289337545633
    DEBUG:root:i=80 residual= 0.0018882564036175609
    DEBUG:root:i=81 residual= 0.0018296840135008097
    DEBUG:root:i=82 residual= 0.0017739603063091636
    DEBUG:root:i=83 residual= 0.0017209481447935104
    DEBUG:root:i=84 residual= 0.0016703980509191751
    DEBUG:root:i=85 residual= 0.001622220384888351
    DEBUG:root:i=86 residual= 0.0015762347029522061
    DEBUG:root:i=87 residual= 0.001532317022792995
    DEBUG:root:i=88 residual= 0.0014903449919074774
    DEBUG:root:i=89 residual= 0.001450187643058598
    DEBUG:root:i=90 residual= 0.0014117526588961482
    DEBUG:root:i=91 residual= 0.0013749193167313933
    DEBUG:root:i=92 residual= 0.001339636160992086
    DEBUG:root:i=93 residual= 0.0013057600008323789
    DEBUG:root:i=94 residual= 0.0012732340255752206
    DEBUG:root:i=95 residual= 0.00124200782738626
    DEBUG:root:i=96 residual= 0.0012120039900764823
    DEBUG:root:i=97 residual= 0.0011831241426989436
    DEBUG:root:i=98 residual= 0.0011553304502740502
    DEBUG:root:i=99 residual= 0.0011285834480077028
    DEBUG:root:i=100 residual= 0.0011027933796867728
    DEBUG:root:i=101 residual= 0.0010779198491945863
    DEBUG:root:i=102 residual= 0.0010539509821683168
    DEBUG:root:i=103 residual= 0.001030795625410974
    DEBUG:root:i=104 residual= 0.0010084446985274553
    DEBUG:root:i=105 residual= 0.000986828119494021
    DEBUG:root:i=106 residual= 0.0009659695788286626
    DEBUG:root:i=107 residual= 0.0009457594715058804
    DEBUG:root:i=108 residual= 0.0009262312669306993
    DEBUG:root:i=109 residual= 0.0009073129622265697
    DEBUG:root:i=110 residual= 0.0008889834280125797
    DEBUG:root:i=111 residual= 0.000871218100655824
    DEBUG:root:i=112 residual= 0.0008540077833458781
    DEBUG:root:i=113 residual= 0.0008373131277039647
    DEBUG:root:i=114 residual= 0.0008211042149923742
    DEBUG:root:i=115 residual= 0.0008053809287957847
    DEBUG:root:i=116 residual= 0.0007901105564087629
    DEBUG:root:i=117 residual= 0.0007752724341116846
    DEBUG:root:i=118 residual= 0.0007608550949953496
    DEBUG:root:i=119 residual= 0.0007468339172191918
    DEBUG:root:i=120 residual= 0.0007332056411541998
    DEBUG:root:i=121 residual= 0.0007199273095466197
    DEBUG:root:i=122 residual= 0.0007070465944707394
    DEBUG:root:i=123 residual= 0.0006944661145098507
    DEBUG:root:i=124 residual= 0.0006822255672886968
    DEBUG:root:i=125 residual= 0.0006702926475554705
    DEBUG:root:i=126 residual= 0.0006586693925783038
    DEBUG:root:i=127 residual= 0.0006473505636677146
    DEBUG:root:i=128 residual= 0.0006362895364873111
    DEBUG:root:i=129 residual= 0.0006255150656215847
    DEBUG:root:i=130 residual= 0.0006149953696876764
    DEBUG:root:i=131 residual= 0.0006047352799214423
    DEBUG:root:i=132 residual= 0.0005947031895630062
    DEBUG:root:i=133 residual= 0.0005849369917996228
    DEBUG:root:i=134 residual= 0.0005753604345954955
    DEBUG:root:i=135 residual= 0.0005660264287143946
    DEBUG:root:i=136 residual= 0.0005569031345658004
    DEBUG:root:i=137 residual= 0.0005479725659824908
    DEBUG:root:i=138 residual= 0.0005392494495026767
    DEBUG:root:i=139 residual= 0.0005307153915055096
    DEBUG:root:i=140 residual= 0.0005223833722993731
    DEBUG:root:i=141 residual= 0.0005142007721588016
    DEBUG:root:i=142 residual= 0.0005062153795734048
    DEBUG:root:i=143 residual= 0.0004983833059668541
    DEBUG:root:i=144 residual= 0.0004907353431917727
    DEBUG:root:i=145 residual= 0.00048323607188649476
    DEBUG:root:i=146 residual= 0.00047587460721842945
    DEBUG:root:i=147 residual= 0.00046868945355527103
    DEBUG:root:i=148 residual= 0.0004616401274688542
    DEBUG:root:i=149 residual= 0.000454726570751518
    DEBUG:root:i=150 residual= 0.0004479595518205315
    DEBUG:root:i=151 residual= 0.0004413192509673536
    DEBUG:root:i=152 residual= 0.0004348020884208381
    DEBUG:root:i=153 residual= 0.0004284268361516297
    DEBUG:root:i=154 residual= 0.00042216284782625735
    DEBUG:root:i=155 residual= 0.0004160113458056003
    DEBUG:root:i=156 residual= 0.0004099986399523914
    DEBUG:root:i=157 residual= 0.0004040776693727821
    DEBUG:root:i=158 residual= 0.000398272299207747
    DEBUG:root:i=159 residual= 0.0003925764176528901
    DEBUG:root:i=160 residual= 0.00038698461139574647
    DEBUG:root:i=161 residual= 0.00038148564635775983
    DEBUG:root:i=162 residual= 0.00037610172876156867
    DEBUG:root:i=163 residual= 0.000370807625586167
    DEBUG:root:i=164 residual= 0.00036560403532348573
    DEBUG:root:i=165 residual= 0.0003604943340178579
    DEBUG:root:i=166 residual= 0.0003554749127943069
    DEBUG:root:i=167 residual= 0.0003505419590510428
    DEBUG:root:i=168 residual= 0.00034569238778203726
    DEBUG:root:i=169 residual= 0.0003409333876334131
    DEBUG:root:i=170 residual= 0.00033624935895204544
    DEBUG:root:i=171 residual= 0.00033164763590320945
    DEBUG:root:i=172 residual= 0.00032712644315324724
    DEBUG:root:i=173 residual= 0.00032267087954096496
    DEBUG:root:i=174 residual= 0.00031830446096137166
    DEBUG:root:i=175 residual= 0.00031399945146404207
    DEBUG:root:i=176 residual= 0.00030977281858213246
    DEBUG:root:i=177 residual= 0.00030561332823708653
    DEBUG:root:i=178 residual= 0.00030151280225254595
    DEBUG:root:i=179 residual= 0.0002974859962705523
    DEBUG:root:i=180 residual= 0.00029353779973462224
    DEBUG:root:i=181 residual= 0.00028962831129319966
    DEBUG:root:i=182 residual= 0.0002858027000911534
    DEBUG:root:i=183 residual= 0.0002820193476509303
    DEBUG:root:i=184 residual= 0.0002783044474199414
    DEBUG:root:i=185 residual= 0.00027465083985589445
    DEBUG:root:i=186 residual= 0.0002710577682591975
    DEBUG:root:i=187 residual= 0.00026749909739010036
    DEBUG:root:i=188 residual= 0.0002640172024257481
    DEBUG:root:i=189 residual= 0.00026058562798425555
    DEBUG:root:i=190 residual= 0.0002572025987319648
    DEBUG:root:i=191 residual= 0.00025387181085534394
    DEBUG:root:i=192 residual= 0.0002505962329450995
    DEBUG:root:i=193 residual= 0.00024736582417972386
    DEBUG:root:i=194 residual= 0.00024418008979409933
    DEBUG:root:i=195 residual= 0.00024105659394990653
    DEBUG:root:i=196 residual= 0.0002379734069108963
    DEBUG:root:i=197 residual= 0.00023492673062719405
    DEBUG:root:i=198 residual= 0.0002319232007721439
    DEBUG:root:i=199 residual= 0.0002289822296006605
    DEBUG:root:i=200 residual= 0.00022608354629483074
    DEBUG:root:i=201 residual= 0.00022320669086184353
    DEBUG:root:i=202 residual= 0.00022039131727069616
    DEBUG:root:i=203 residual= 0.00021760488743893802
    DEBUG:root:i=204 residual= 0.00021486762852873653
    DEBUG:root:i=205 residual= 0.00021216023014858365
    DEBUG:root:i=206 residual= 0.0002095070231007412
    DEBUG:root:i=207 residual= 0.00020687162759713829
    DEBUG:root:i=208 residual= 0.00020429094729479402
    DEBUG:root:i=209 residual= 0.00020173964730929583
    DEBUG:root:i=210 residual= 0.00019922912179026753
    DEBUG:root:i=211 residual= 0.00019674820941872895
    DEBUG:root:i=212 residual= 0.00019430898828431964
    DEBUG:root:i=213 residual= 0.00019189741578884423
    DEBUG:root:i=214 residual= 0.00018953224935103208
    DEBUG:root:i=215 residual= 0.00018718236242420971
    DEBUG:root:i=216 residual= 0.00018488330533728004
    DEBUG:root:i=217 residual= 0.00018259685020893812
    DEBUG:root:i=218 residual= 0.00018035637913271785
    DEBUG:root:i=219 residual= 0.0001781344471964985
    DEBUG:root:i=220 residual= 0.00017595991084817797
    DEBUG:root:i=221 residual= 0.0001738045975798741
    DEBUG:root:i=222 residual= 0.00017168137128464878
    DEBUG:root:i=223 residual= 0.0001695916726021096
    DEBUG:root:i=224 residual= 0.00016752371448092163
    DEBUG:root:i=225 residual= 0.00016547732229810208
    DEBUG:root:i=226 residual= 0.00016347414930351079
    DEBUG:root:i=227 residual= 0.00016149050497915596
    DEBUG:root:i=228 residual= 0.0001595266949152574
    DEBUG:root:i=229 residual= 0.00015761115355417132
    DEBUG:root:i=230 residual= 0.00015569926472380757
    DEBUG:root:i=231 residual= 0.00015381522825919092
    DEBUG:root:i=232 residual= 0.0001519646611995995
    DEBUG:root:i=233 residual= 0.0001501374936196953
    DEBUG:root:i=234 residual= 0.00014832912711426616
    DEBUG:root:i=235 residual= 0.00014655699487775564
    DEBUG:root:i=236 residual= 0.0001447917165933177
    DEBUG:root:i=237 residual= 0.00014305816148407757
    DEBUG:root:i=238 residual= 0.00014135229866951704
    DEBUG:root:i=239 residual= 0.00013965395919512957
    DEBUG:root:i=240 residual= 0.00013798920554108918
    DEBUG:root:i=241 residual= 0.00013635153300128877
    DEBUG:root:i=242 residual= 0.0001347241923213005
    DEBUG:root:i=243 residual= 0.00013312854571267962
    DEBUG:root:i=244 residual= 0.00013153397594578564
    DEBUG:root:i=245 residual= 0.00012997658632230014
    DEBUG:root:i=246 residual= 0.00012844157754443586
    DEBUG:root:i=247 residual= 0.00012691464507952332
    DEBUG:root:i=248 residual= 0.0001254083326784894
    DEBUG:root:i=249 residual= 0.00012393738143146038
    DEBUG:root:i=250 residual= 0.00012247191625647247
    DEBUG:root:i=251 residual= 0.00012102329492336139
    DEBUG:root:i=252 residual= 0.00011960278789047152
    DEBUG:root:i=253 residual= 0.0001181889238068834
    DEBUG:root:i=254 residual= 0.00011680227180477232
    DEBUG:root:i=255 residual= 0.00011543372966116294
    DEBUG:root:i=256 residual= 0.00011406873090891168
    DEBUG:root:i=257 residual= 0.00011273782001808286
    DEBUG:root:i=258 residual= 0.00011141248978674412
    DEBUG:root:i=259 residual= 0.00011011026072083041
    DEBUG:root:i=260 residual= 0.00010882852802751586
    DEBUG:root:i=261 residual= 0.00010754974209703505
    DEBUG:root:i=262 residual= 0.00010629312600940466
    DEBUG:root:i=263 residual= 0.00010505628597456962
    DEBUG:root:i=264 residual= 0.00010383311018813401
    DEBUG:root:i=265 residual= 0.00010262461728416383
    DEBUG:root:i=266 residual= 0.00010143294639419764
    DEBUG:root:i=267 residual= 0.00010024397488450631
    DEBUG:root:i=268 residual= 9.908818901749328e-05
    DEBUG:root:i=269 residual= 9.793486970011145e-05
    DEBUG:root:i=270 residual= 9.680299990577623e-05
    DEBUG:root:i=271 residual= 9.567133383825421e-05
    DEBUG:root:i=272 residual= 9.45757346926257e-05
    DEBUG:root:i=273 residual= 9.347673039883375e-05
    DEBUG:root:i=274 residual= 9.239432984031737e-05
    DEBUG:root:i=275 residual= 9.13231197046116e-05
    DEBUG:root:i=276 residual= 9.027804480865598e-05
    DEBUG:root:i=277 residual= 8.923355198930949e-05
    DEBUG:root:i=278 residual= 8.820158109301701e-05
    DEBUG:root:i=279 residual= 8.719314791960642e-05
    DEBUG:root:i=280 residual= 8.619062282377854e-05
    DEBUG:root:i=281 residual= 8.519322727806866e-05
    DEBUG:root:i=282 residual= 8.420868107350543e-05
    DEBUG:root:i=283 residual= 8.325254020746797e-05
    DEBUG:root:i=284 residual= 8.230250386986881e-05
    DEBUG:root:i=285 residual= 8.134787640301511e-05
    DEBUG:root:i=286 residual= 8.041330147534609e-05
    DEBUG:root:i=287 residual= 7.948822167236358e-05
    DEBUG:root:i=288 residual= 7.859462493797764e-05
    DEBUG:root:i=289 residual= 7.768834620947018e-05
    DEBUG:root:i=290 residual= 7.680062117287889e-05
    DEBUG:root:i=291 residual= 7.592132897116244e-05
    DEBUG:root:i=292 residual= 7.504998939111829e-05
    DEBUG:root:i=293 residual= 7.419746543746442e-05
    DEBUG:root:i=294 residual= 7.334975089179352e-05
    DEBUG:root:i=295 residual= 7.25165446056053e-05
    DEBUG:root:i=296 residual= 7.168885349528864e-05
    DEBUG:root:i=297 residual= 7.087658741511405e-05
    DEBUG:root:i=298 residual= 7.006428495515138e-05
    DEBUG:root:i=299 residual= 6.92674148012884e-05
    DEBUG:root:i=300 residual= 6.8488901888486e-05
    DEBUG:root:i=301 residual= 6.770378240616992e-05
    DEBUG:root:i=302 residual= 6.69300279696472e-05
    DEBUG:root:i=303 residual= 6.618213228648528e-05
    DEBUG:root:i=304 residual= 6.542012852150947e-05
    DEBUG:root:i=305 residual= 6.468951323768124e-05
    DEBUG:root:i=306 residual= 6.394870433723554e-05
    DEBUG:root:i=307 residual= 6.322687841020525e-05
    DEBUG:root:i=308 residual= 6.251379090826958e-05
    DEBUG:root:i=309 residual= 6.180552736623213e-05
    DEBUG:root:i=310 residual= 6.110957474447787e-05
    DEBUG:root:i=311 residual= 6.0408525314414874e-05
    DEBUG:root:i=312 residual= 5.973687439109199e-05
    DEBUG:root:i=313 residual= 5.905028592678718e-05
    DEBUG:root:i=314 residual= 5.839604637003504e-05
    DEBUG:root:i=315 residual= 5.772921576863155e-05
    DEBUG:root:i=316 residual= 5.707934178644791e-05
    DEBUG:root:i=317 residual= 5.6448119721608236e-05
    DEBUG:root:i=318 residual= 5.579555727308616e-05
    DEBUG:root:i=319 residual= 5.517104727914557e-05
    DEBUG:root:i=320 residual= 5.455843347590417e-05
    DEBUG:root:i=321 residual= 5.394232357502915e-05
    DEBUG:root:i=322 residual= 5.332994624041021e-05
    DEBUG:root:i=323 residual= 5.273174974718131e-05
    DEBUG:root:i=324 residual= 5.214187331148423e-05
    DEBUG:root:i=325 residual= 5.155721373739652e-05
    DEBUG:root:i=326 residual= 5.096870518173091e-05
    DEBUG:root:i=327 residual= 5.040755058871582e-05
    DEBUG:root:i=328 residual= 4.984297265764326e-05
    DEBUG:root:i=329 residual= 4.926495239487849e-05
    DEBUG:root:i=330 residual= 4.8724123189458624e-05
    DEBUG:root:i=331 residual= 4.817700028070249e-05
    DEBUG:root:i=332 residual= 4.764696495840326e-05
    DEBUG:root:i=333 residual= 4.7098925278987736e-05
    DEBUG:root:i=334 residual= 4.660025660996325e-05
    DEBUG:root:i=335 residual= 4.6052111429162323e-05
    DEBUG:root:i=336 residual= 4.5547509216703475e-05
    DEBUG:root:i=337 residual= 4.503193122218363e-05
    DEBUG:root:i=338 residual= 4.4531840103445575e-05
    DEBUG:root:i=339 residual= 4.4037245970685035e-05
    DEBUG:root:i=340 residual= 4.354004340711981e-05
    DEBUG:root:i=341 residual= 4.305178299546242e-05
    DEBUG:root:i=342 residual= 4.257997352397069e-05
    DEBUG:root:i=343 residual= 4.210381302982569e-05
    DEBUG:root:i=344 residual= 4.163216726738028e-05
    DEBUG:root:i=345 residual= 4.1168383177137e-05
    DEBUG:root:i=346 residual= 4.071654984727502e-05
    DEBUG:root:i=347 residual= 4.0256549254991114e-05
    DEBUG:root:i=348 residual= 3.981372356065549e-05
    DEBUG:root:i=349 residual= 3.9359965740004554e-05
    DEBUG:root:i=350 residual= 3.89261003874708e-05
    DEBUG:root:i=351 residual= 3.849617496598512e-05
    DEBUG:root:i=352 residual= 3.806547829299234e-05
    DEBUG:root:i=353 residual= 3.764238499570638e-05
    DEBUG:root:i=354 residual= 3.722483597812243e-05
    DEBUG:root:i=355 residual= 3.680725785670802e-05
    DEBUG:root:i=356 residual= 3.640410432126373e-05
    DEBUG:root:i=357 residual= 3.600340278353542e-05
    DEBUG:root:i=358 residual= 3.559655669960193e-05
    DEBUG:root:i=359 residual= 3.5205612221034244e-05
    DEBUG:root:i=360 residual= 3.480755185591988e-05
    DEBUG:root:i=361 residual= 3.4419204894220456e-05
    DEBUG:root:i=362 residual= 3.4057160519296303e-05
    DEBUG:root:i=363 residual= 3.366635792190209e-05
    DEBUG:root:i=364 residual= 3.329436731291935e-05
    DEBUG:root:i=365 residual= 3.2916748750722036e-05
    DEBUG:root:i=366 residual= 3.2575640943832695e-05
    DEBUG:root:i=367 residual= 3.21950028592255e-05
    DEBUG:root:i=368 residual= 3.1843264878261834e-05
    DEBUG:root:i=369 residual= 3.1486870284425095e-05
    DEBUG:root:i=370 residual= 3.1148574635153636e-05
    DEBUG:root:i=371 residual= 3.0794184567639604e-05
    DEBUG:root:i=372 residual= 3.045223274966702e-05
    DEBUG:root:i=373 residual= 3.0126049750833772e-05
    DEBUG:root:i=374 residual= 2.978108386741951e-05
    DEBUG:root:i=375 residual= 2.946021231764462e-05
    DEBUG:root:i=376 residual= 2.9135760996723548e-05
    DEBUG:root:i=377 residual= 2.8815118639613502e-05
    DEBUG:root:i=378 residual= 2.8484981157816947e-05
    DEBUG:root:i=379 residual= 2.81868069578195e-05
    DEBUG:root:i=380 residual= 2.7869080440723337e-05
    DEBUG:root:i=381 residual= 2.7556858185562305e-05
    DEBUG:root:i=382 residual= 2.7252572181168944e-05
    DEBUG:root:i=383 residual= 2.6954952772939578e-05
    DEBUG:root:i=384 residual= 2.665684951352887e-05
    DEBUG:root:i=385 residual= 2.6369050829089247e-05
    DEBUG:root:i=386 residual= 2.607192436698824e-05
    DEBUG:root:i=387 residual= 2.5784211175050586e-05
    DEBUG:root:i=388 residual= 2.55060749623226e-05
    DEBUG:root:i=389 residual= 2.5220941097359173e-05
    DEBUG:root:i=390 residual= 2.4944823962869123e-05
    DEBUG:root:i=391 residual= 2.466968544467818e-05
    DEBUG:root:i=392 residual= 2.4391381884925067e-05
    DEBUG:root:i=393 residual= 2.4128483346430585e-05
    DEBUG:root:i=394 residual= 2.3862718080636114e-05
    DEBUG:root:i=395 residual= 2.360632424824871e-05
    DEBUG:root:i=396 residual= 2.333511110919062e-05
    DEBUG:root:i=397 residual= 2.3083379346644506e-05
    DEBUG:root:i=398 residual= 2.28292919928208e-05
    DEBUG:root:i=399 residual= 2.2574851755052805e-05
    DEBUG:root:i=400 residual= 2.2331827494781464e-05
    DEBUG:root:i=401 residual= 2.2084588636062108e-05
    DEBUG:root:i=402 residual= 2.183979631809052e-05
    DEBUG:root:i=403 residual= 2.1608680981444195e-05
    DEBUG:root:i=404 residual= 2.1357494915719144e-05
    DEBUG:root:i=405 residual= 2.112522088282276e-05
    DEBUG:root:i=406 residual= 2.0903155018459074e-05
    DEBUG:root:i=407 residual= 2.0672929167631082e-05
    DEBUG:root:i=408 residual= 2.043424865405541e-05
    DEBUG:root:i=409 residual= 2.022028093051631e-05
    DEBUG:root:i=410 residual= 1.9995923139504157e-05
    DEBUG:root:i=411 residual= 1.9772811356233433e-05
    DEBUG:root:i=412 residual= 1.9557335690478794e-05
    DEBUG:root:i=413 residual= 1.9345250620972365e-05
    DEBUG:root:i=414 residual= 1.9134058675263077e-05
    DEBUG:root:i=415 residual= 1.8920109141618013e-05
    DEBUG:root:i=416 residual= 1.8710941731114872e-05
    DEBUG:root:i=417 residual= 1.8506820197217166e-05
    DEBUG:root:i=418 residual= 1.830958171922248e-05
    DEBUG:root:i=419 residual= 1.810740832297597e-05
    DEBUG:root:i=420 residual= 1.7904611013364047e-05
    DEBUG:root:i=421 residual= 1.7710386600811034e-05
    DEBUG:root:i=422 residual= 1.751956187945325e-05
    DEBUG:root:i=423 residual= 1.7319365724688396e-05
    DEBUG:root:i=424 residual= 1.714008249109611e-05
    DEBUG:root:i=425 residual= 1.6944630260695703e-05
    DEBUG:root:i=426 residual= 1.676342071732506e-05
    DEBUG:root:i=427 residual= 1.6580946976318955e-05
    DEBUG:root:i=428 residual= 1.6391544704674743e-05
    DEBUG:root:i=429 residual= 1.6232719644904137e-05
    DEBUG:root:i=430 residual= 1.60320705617778e-05
    DEBUG:root:i=431 residual= 1.586649341334123e-05
    DEBUG:root:i=432 residual= 1.5681813238188624e-05
    DEBUG:root:i=433 residual= 1.5531517419731244e-05
    DEBUG:root:i=434 residual= 1.5348176020779647e-05
    DEBUG:root:i=435 residual= 1.5179334695858415e-05
    DEBUG:root:i=436 residual= 1.5014930795587134e-05
    DEBUG:root:i=437 residual= 1.4850589650450274e-05
    DEBUG:root:i=438 residual= 1.4689120689581614e-05
    DEBUG:root:i=439 residual= 1.4527646271744743e-05
    DEBUG:root:i=440 residual= 1.4381190339918248e-05
    DEBUG:root:i=441 residual= 1.4209340406523552e-05
    DEBUG:root:i=442 residual= 1.4056428881303873e-05
    DEBUG:root:i=443 residual= 1.389891895087203e-05
    DEBUG:root:i=444 residual= 1.3753830899077002e-05
    DEBUG:root:i=445 residual= 1.3604428204416763e-05
    DEBUG:root:i=446 residual= 1.3452286111714784e-05
    DEBUG:root:i=447 residual= 1.3304531421454158e-05
    DEBUG:root:i=448 residual= 1.316492216574261e-05
    DEBUG:root:i=449 residual= 1.3015472177357879e-05
    DEBUG:root:i=450 residual= 1.2869563761341851e-05
    DEBUG:root:i=451 residual= 1.2739201338263229e-05
    DEBUG:root:i=452 residual= 1.2593224710144568e-05
    DEBUG:root:i=453 residual= 1.245771454705391e-05
    DEBUG:root:i=454 residual= 1.2318892913754098e-05
    DEBUG:root:i=455 residual= 1.2194427654321771e-05
    DEBUG:root:i=456 residual= 1.2054858416377101e-05
    DEBUG:root:i=457 residual= 1.1922947123821359e-05
    DEBUG:root:i=458 residual= 1.179051923827501e-05
    DEBUG:root:i=459 residual= 1.1657328286673874e-05
    DEBUG:root:i=460 residual= 1.1535621524672024e-05
    DEBUG:root:i=461 residual= 1.1412051208026242e-05
    DEBUG:root:i=462 residual= 1.1283730600553099e-05
    DEBUG:root:i=463 residual= 1.116690236813156e-05
    DEBUG:root:i=464 residual= 1.1040708159271162e-05
    DEBUG:root:i=465 residual= 1.0926914910669439e-05
    DEBUG:root:i=466 residual= 1.0800821655720938e-05
    DEBUG:root:i=467 residual= 1.0684790140658151e-05
    DEBUG:root:i=468 residual= 1.056322616932448e-05
    DEBUG:root:i=469 residual= 1.0451340131112374e-05
    DEBUG:root:i=470 residual= 1.034284014167497e-05
    DEBUG:root:i=471 residual= 1.0225820915366057e-05
    DEBUG:root:i=472 residual= 1.0119668331753928e-05
    DEBUG:root:i=473 residual= 1.0003052011597902e-05
    DEBUG:root:i=474 residual= 9.89186082733795e-06
    DEBUG:root:i=475 residual= 9.78262414719211e-06
    DEBUG:root:i=476 residual= 9.690269507700577e-06
    DEBUG:root:i=477 residual= 9.56998974288581e-06
    DEBUG:root:i=478 residual= 9.468605639995076e-06
    DEBUG:root:i=479 residual= 9.36931883188663e-06
    DEBUG:root:i=480 residual= 9.268116627936251e-06
    DEBUG:root:i=481 residual= 9.161152775050141e-06
    DEBUG:root:i=482 residual= 9.070800842891913e-06
    DEBUG:root:i=483 residual= 8.961502317106351e-06
    DEBUG:root:i=484 residual= 8.873053047864232e-06
    DEBUG:root:i=485 residual= 8.77448110259138e-06
    DEBUG:root:i=486 residual= 8.675451681483537e-06
    DEBUG:root:i=487 residual= 8.586973308410961e-06
    DEBUG:root:i=488 residual= 8.490532309224363e-06
    DEBUG:root:i=489 residual= 8.394657015742268e-06
    DEBUG:root:i=490 residual= 8.302999049192294e-06
    DEBUG:root:i=491 residual= 8.221156349463854e-06
    DEBUG:root:i=492 residual= 8.125264685077127e-06
    DEBUG:root:i=493 residual= 8.039784006541595e-06
    DEBUG:root:i=494 residual= 7.944486242195126e-06
    DEBUG:root:i=495 residual= 7.873053618823178e-06
    DEBUG:root:i=496 residual= 7.77926470618695e-06
    DEBUG:root:i=497 residual= 7.68679274187889e-06
    DEBUG:root:i=498 residual= 7.619046300533228e-06
    DEBUG:root:i=499 residual= 7.531978098995751e-06
    DEBUG:root:i=500 residual= 7.441050911438651e-06
    DEBUG:root:i=501 residual= 7.361990810750285e-06
    DEBUG:root:i=502 residual= 7.283066679519834e-06
    DEBUG:root:i=503 residual= 7.205330348369898e-06
    DEBUG:root:i=504 residual= 7.1284271143667866e-06
    DEBUG:root:i=505 residual= 7.047851795505267e-06
    DEBUG:root:i=506 residual= 6.9760808401042596e-06
    DEBUG:root:i=507 residual= 6.8954382186348084e-06
    DEBUG:root:i=508 residual= 6.819973350502551e-06
    DEBUG:root:i=509 residual= 6.743743597326102e-06
    DEBUG:root:i=510 residual= 6.6791790231945924e-06
    DEBUG:root:i=511 residual= 6.604365808016155e-06
    DEBUG:root:i=512 residual= 6.5304157033097e-06
    DEBUG:root:i=513 residual= 6.45491900286288e-06
    DEBUG:root:i=514 residual= 6.385832421074156e-06
    DEBUG:root:i=515 residual= 6.3293141465692315e-06
    DEBUG:root:i=516 residual= 6.25007169219316e-06
    DEBUG:root:i=517 residual= 6.180852778925328e-06
    DEBUG:root:i=518 residual= 6.116134045441868e-06
    DEBUG:root:i=519 residual= 6.045021564204944e-06
    DEBUG:root:i=520 residual= 5.991940724925371e-06
    DEBUG:root:i=521 residual= 5.9211924963165075e-06
    DEBUG:root:i=522 residual= 5.8509631344350055e-06
    DEBUG:root:i=523 residual= 5.796907771582482e-06
    DEBUG:root:i=524 residual= 5.730364591727266e-06
    DEBUG:root:i=525 residual= 5.6632779887877405e-06
    DEBUG:root:i=526 residual= 5.602880264632404e-06
    DEBUG:root:i=527 residual= 5.544397026824299e-06
    DEBUG:root:i=528 residual= 5.478648290591082e-06
    DEBUG:root:i=529 residual= 5.4229185479925945e-06
    DEBUG:root:i=530 residual= 5.375333785195835e-06
    DEBUG:root:i=531 residual= 5.302251793182222e-06
    DEBUG:root:i=532 residual= 5.252323717286345e-06
    DEBUG:root:i=533 residual= 5.188186605664669e-06
    DEBUG:root:i=534 residual= 5.132432306709234e-06
    DEBUG:root:i=535 residual= 5.0893318075395655e-06
    DEBUG:root:i=536 residual= 5.02226066600997e-06
    DEBUG:root:i=537 residual= 4.969564088241896e-06
    DEBUG:root:i=538 residual= 4.913400061923312e-06
    DEBUG:root:i=539 residual= 4.862615242018364e-06
    DEBUG:root:i=540 residual= 4.816935415874468e-06
    DEBUG:root:i=541 residual= 4.755795998789836e-06
    DEBUG:root:i=542 residual= 4.706311756308423e-06
    DEBUG:root:i=543 residual= 4.6502686927851755e-06
    DEBUG:root:i=544 residual= 4.610442829289241e-06
    DEBUG:root:i=545 residual= 4.560955403576372e-06
    DEBUG:root:i=546 residual= 4.500430804910138e-06
    DEBUG:root:i=547 residual= 4.45308796770405e-06
    DEBUG:root:i=548 residual= 4.4221546886547e-06
    DEBUG:root:i=549 residual= 4.358222668088274e-06
    DEBUG:root:i=550 residual= 4.3096843000967056e-06
    DEBUG:root:i=551 residual= 4.2652136471588165e-06
    DEBUG:root:i=552 residual= 4.224138137942646e-06
    DEBUG:root:i=553 residual= 4.170623014942976e-06
    DEBUG:root:i=554 residual= 4.145331331528723e-06
    DEBUG:root:i=555 residual= 4.079966402059654e-06
    DEBUG:root:i=556 residual= 4.035824531456456e-06
    DEBUG:root:i=557 residual= 4.005676146334736e-06
    DEBUG:root:i=558 residual= 3.954736712330487e-06
    DEBUG:root:i=559 residual= 3.906752226612298e-06
    DEBUG:root:i=560 residual= 3.864045993395848e-06
    DEBUG:root:i=561 residual= 3.828268290817505e-06
    DEBUG:root:i=562 residual= 3.7866188904445153e-06
    DEBUG:root:i=563 residual= 3.740935426321812e-06
    DEBUG:root:i=564 residual= 3.7030231396784075e-06
    DEBUG:root:i=565 residual= 3.6599283248506254e-06
    DEBUG:root:i=566 residual= 3.6192127481626812e-06
    DEBUG:root:i=567 residual= 3.5817804473481374e-06
    DEBUG:root:i=568 residual= 3.5455716442811536e-06
    DEBUG:root:i=569 residual= 3.5146060781698907e-06
    DEBUG:root:i=570 residual= 3.467127044132212e-06
    DEBUG:root:i=571 residual= 3.4297800084459595e-06
    DEBUG:root:i=572 residual= 3.394441819182248e-06
    DEBUG:root:i=573 residual= 3.35284721586504e-06
    DEBUG:root:i=574 residual= 3.322558086438221e-06
    DEBUG:root:i=575 residual= 3.2803341127873864e-06
    DEBUG:root:i=576 residual= 3.247938593631261e-06
    DEBUG:root:i=577 residual= 3.231232994949096e-06
    DEBUG:root:i=578 residual= 3.1773740829521557e-06
    DEBUG:root:i=579 residual= 3.1391814445669297e-06
    DEBUG:root:i=580 residual= 3.113679895250243e-06
    DEBUG:root:i=581 residual= 3.0762698770558927e-06
    DEBUG:root:i=582 residual= 3.0393298402486835e-06
    DEBUG:root:i=583 residual= 3.023300223503611e-06
    DEBUG:root:i=584 residual= 2.97307792607171e-06
    DEBUG:root:i=585 residual= 2.9517609618778806e-06
    DEBUG:root:i=586 residual= 2.911436013164348e-06
    DEBUG:root:i=587 residual= 2.888628387154313e-06
    DEBUG:root:i=588 residual= 2.8469360131566646e-06
    DEBUG:root:i=589 residual= 2.8310143989074277e-06
    DEBUG:root:i=590 residual= 2.788082838378614e-06
    DEBUG:root:i=591 residual= 2.7557841804082273e-06
    DEBUG:root:i=592 residual= 2.7275204956822563e-06
    DEBUG:root:i=593 residual= 2.69830502475088e-06
    DEBUG:root:i=594 residual= 2.6717714263213566e-06
    DEBUG:root:i=595 residual= 2.6402224193589063e-06
    DEBUG:root:i=596 residual= 2.6218363018415403e-06
    DEBUG:root:i=597 residual= 2.582552497187862e-06
    DEBUG:root:i=598 residual= 2.5551923954481026e-06
    DEBUG:root:i=599 residual= 2.548667453083908e-06
    DEBUG:root:i=600 residual= 2.5000827008625492e-06
    DEBUG:root:i=601 residual= 2.4723910883039935e-06
    DEBUG:root:i=602 residual= 2.4541861876059556e-06
    DEBUG:root:i=603 residual= 2.419029669908923e-06
    DEBUG:root:i=604 residual= 2.401731535428553e-06
    DEBUG:root:i=605 residual= 2.36715186474612e-06
    DEBUG:root:i=606 residual= 2.3409506866300944e-06
    DEBUG:root:i=607 residual= 2.332454414499807e-06
    DEBUG:root:i=608 residual= 2.29271199714276e-06
    DEBUG:root:i=609 residual= 2.265669081680244e-06
    DEBUG:root:i=610 residual= 2.2421440917241853e-06
    DEBUG:root:i=611 residual= 2.219579528173199e-06
    DEBUG:root:i=612 residual= 2.1988832941133296e-06
    DEBUG:root:i=613 residual= 2.1689893401344307e-06
    DEBUG:root:i=614 residual= 2.1467490114446264e-06
    DEBUG:root:i=615 residual= 2.1367902718338883e-06
    DEBUG:root:i=616 residual= 2.099803623423213e-06
    DEBUG:root:i=617 residual= 2.0770116861967836e-06
    DEBUG:root:i=618 residual= 2.0614891127479495e-06
    DEBUG:root:i=619 residual= 2.042681671809987e-06
    DEBUG:root:i=620 residual= 2.0108450371481013e-06
    DEBUG:root:i=621 residual= 1.9919179976568557e-06
    DEBUG:root:i=622 residual= 1.9662627437355695e-06
    DEBUG:root:i=623 residual= 1.948846374943969e-06
    DEBUG:root:i=624 residual= 1.929242444020929e-06
    DEBUG:root:i=625 residual= 1.911831077450188e-06
    DEBUG:root:i=626 residual= 1.8830486396836932e-06
    DEBUG:root:i=627 residual= 1.8632129012985388e-06
    DEBUG:root:i=628 residual= 1.8450025436322903e-06
    DEBUG:root:i=629 residual= 1.8230831528853741e-06
    DEBUG:root:i=630 residual= 1.802846668397251e-06
    DEBUG:root:i=631 residual= 1.786728262231918e-06
    DEBUG:root:i=632 residual= 1.7654791690802085e-06
    DEBUG:root:i=633 residual= 1.7564034351380542e-06
    DEBUG:root:i=634 residual= 1.7269475165448966e-06
    DEBUG:root:i=635 residual= 1.7115370383180561e-06
    DEBUG:root:i=636 residual= 1.6905922848309274e-06
    DEBUG:root:i=637 residual= 1.6712757542336476e-06
    DEBUG:root:i=638 residual= 1.6610698594377027e-06
    DEBUG:root:i=639 residual= 1.6419478470197646e-06
    DEBUG:root:i=640 residual= 1.6185374533961294e-06
    DEBUG:root:i=641 residual= 1.5996926094885566e-06
    DEBUG:root:i=642 residual= 1.5828546793272835e-06
    DEBUG:root:i=643 residual= 1.5690202417317778e-06
    DEBUG:root:i=644 residual= 1.5492040574827115e-06
    DEBUG:root:i=645 residual= 1.552035200802493e-06
    DEBUG:root:i=646 residual= 1.5146667919907486e-06
    DEBUG:root:i=647 residual= 1.4983457958805957e-06
    DEBUG:root:i=648 residual= 1.4829050769549212e-06
    DEBUG:root:i=649 residual= 1.4664248055851203e-06
    DEBUG:root:i=650 residual= 1.4519442856908427e-06
    DEBUG:root:i=651 residual= 1.4336345657284255e-06
    DEBUG:root:i=652 residual= 1.4488643955701264e-06
    DEBUG:root:i=653 residual= 1.4032285662324284e-06
    DEBUG:root:i=654 residual= 1.388099803989462e-06
    DEBUG:root:i=655 residual= 1.3852543361281278e-06
    DEBUG:root:i=656 residual= 1.3686340025742538e-06
    DEBUG:root:i=657 residual= 1.346629119325371e-06
    DEBUG:root:i=658 residual= 1.3355291912375833e-06
    DEBUG:root:i=659 residual= 1.3164637948648306e-06
    DEBUG:root:i=660 residual= 1.300701342188404e-06
    DEBUG:root:i=661 residual= 1.2939151474711252e-06
    DEBUG:root:i=662 residual= 1.2797586350643542e-06
    DEBUG:root:i=663 residual= 1.2667846931435633e-06
    DEBUG:root:i=664 residual= 1.2637782447200152e-06
    DEBUG:root:i=665 residual= 1.240438109562092e-06
    DEBUG:root:i=666 residual= 1.2457549019018188e-06
    DEBUG:root:i=667 residual= 1.2070589718859992e-06
    DEBUG:root:i=668 residual= 1.1935401289520087e-06
    DEBUG:root:i=669 residual= 1.1790356211349717e-06
    DEBUG:root:i=670 residual= 1.1664535577438073e-06
    DEBUG:root:i=671 residual= 1.1612204389166436e-06
    DEBUG:root:i=672 residual= 1.143740746556432e-06
    DEBUG:root:i=673 residual= 1.141504071711097e-06
    DEBUG:root:i=674 residual= 1.1169037179570296e-06
    DEBUG:root:i=675 residual= 1.1046656709368108e-06
    DEBUG:root:i=676 residual= 1.0922068440777366e-06
    DEBUG:root:i=677 residual= 1.1018149734809413e-06
    DEBUG:root:i=678 residual= 1.0713874871726148e-06
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

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   The assignment is worth 8 points.
   1. There are 6 parts to the output above.  (4 in Task1 and 2 in Task2.)
   1. Each part that you get incorrect will result in -2 points.  (But you cannot go negative.)
   1. Another way of phrasing this is that the first 2 parts you complete are not worth any points,
      but each part after that is worth 2 points.
